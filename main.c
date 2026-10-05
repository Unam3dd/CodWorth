#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#include "llama.h"

static llama_token sample_token(const float * logits, int32_t n_vocab, float temperature) {
    if (temperature <= 0.0f) {
        llama_token best = 0;
        float best_logit = -FLT_MAX;
        for (int32_t i = 0; i < n_vocab; ++i) {
            if (logits[i] > best_logit) {
                best_logit = logits[i];
                best = i;
            }
        }
        return best;
    }

    float max_logit = -FLT_MAX;
    for (int32_t i = 0; i < n_vocab; ++i) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }

    float * probs = malloc((size_t) n_vocab * sizeof(float));
    if (probs == NULL) {
        return 0;
    }

    float sum = 0.0f;
    for (int32_t i = 0; i < n_vocab; ++i) {
        probs[i] = expf((logits[i] - max_logit) / temperature);
        sum += probs[i];
    }

    float threshold = ((float) rand() / (float) RAND_MAX) * sum;
    float cdf = 0.0f;
    llama_token sampled = n_vocab - 1;
    for (int32_t i = 0; i < n_vocab; ++i) {
        cdf += probs[i];
        if (cdf >= threshold) {
            sampled = i;
            break;
        }
    }

    free(probs);
    return sampled;
}

static bool print_token_piece(const struct llama_vocab * vocab, llama_token token) {
    char piece[256];
    int32_t n = llama_token_to_piece(vocab, token, piece, (int32_t) sizeof(piece), 0, true);
    if (n < 0) {
        int32_t need = -n;
        char * dyn = malloc((size_t) need);
        if (dyn == NULL) {
            return false;
        }
        n = llama_token_to_piece(vocab, token, dyn, need, 0, true);
        if (n > 0) {
            (void) fwrite(dyn, 1, (size_t) n, stdout);
            (void) fflush(stdout);
        }
        free(dyn);
        return n > 0;
    }

    if (n > 0) {
        (void) fwrite(piece, 1, (size_t) n, stdout);
        (void) fflush(stdout);
    }
    return n >= 0;
}

int main(void) {
    llama_backend_init();
    srand((unsigned int) time(NULL));

    struct llama_model_params model_params = llama_model_default_params();
    const char * model_path = "llama-3.1-8b-instruct-q4_k_m.gguf";
    struct llama_model * model = llama_load_model_from_file(model_path, model_params);
    if (model == NULL) {
        fprintf(stderr, "Error: Could not load the Codsworth model file.\n");
        llama_backend_free();
        return 1;
    }

    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    struct llama_context * ctx = llama_new_context_with_model(model, ctx_params);
    if (ctx == NULL) {
        fprintf(stderr, "Error: Could not create llama context.\n");
        llama_free_model(model);
        llama_backend_free();
        return 1;
    }

    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_ctx = llama_n_ctx(ctx);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const llama_token eos = llama_token_eos(vocab);

    struct llama_batch batch = llama_batch_init(1, 0, 1);

    const char * system_prompt =
        "System: You are Codsworth, the loyal robot butler from Fallout. "
        "Address the user as 'Monsieur' or 'Madame' with extreme British politeness. "
        "You are also an expert software engineer proficient in modern C and C++.\n";

    printf("Codsworth: À votre service, Monsieur ! Que puis-je faire pour vous ?\n");

    char user_input[1024];
    int32_t n_past = 0;
    const float temperature = 0.8f;
    const int32_t max_gen_tokens = 256;

    while (true) {
        printf("\nVous: ");
        if (fgets(user_input, sizeof(user_input), stdin) == NULL) {
            break;
        }
        user_input[strcspn(user_input, "\n")] = '\0';

        if (strcmp(user_input, "quitter") == 0) {
            break;
        }

        char prompt[4096];
        int prompt_len = snprintf(
            prompt,
            sizeof(prompt),
            "%sUser: %s\nAssistant:",
            system_prompt,
            user_input
        );
        if (prompt_len < 0 || prompt_len >= (int) sizeof(prompt)) {
            fprintf(stderr, "Error: Prompt is too long.\n");
            continue;
        }

        int32_t max_prompt_tokens = (int32_t) strlen(prompt) + 8;
        llama_token * prompt_tokens = malloc((size_t) max_prompt_tokens * sizeof(llama_token));
        if (prompt_tokens == NULL) {
            fprintf(stderr, "Error: Failed to allocate prompt tokens.\n");
            break;
        }

        int32_t n_prompt_tokens = llama_tokenize(
            vocab,
            prompt,
            (int32_t) strlen(prompt),
            prompt_tokens,
            max_prompt_tokens,
            true,
            true
        );
        if (n_prompt_tokens < 0) {
            int32_t needed = -n_prompt_tokens;
            llama_token * resized = realloc(prompt_tokens, (size_t) needed * sizeof(llama_token));
            if (resized == NULL) {
                free(prompt_tokens);
                fprintf(stderr, "Error: Failed to resize prompt token buffer.\n");
                continue;
            }
            prompt_tokens = resized;
            n_prompt_tokens = llama_tokenize(
                vocab,
                prompt,
                (int32_t) strlen(prompt),
                prompt_tokens,
                needed,
                true,
                true
            );
        }
        if (n_prompt_tokens <= 0) {
            free(prompt_tokens);
            continue;
        }

        if (n_past + n_prompt_tokens + max_gen_tokens >= n_ctx) {
            llama_kv_cache_clear(ctx);
            n_past = 0;
        }

        for (int32_t i = 0; i < n_prompt_tokens; ++i) {
            batch.n_tokens = 1;
            batch.token[0] = prompt_tokens[i];
            batch.pos[0] = n_past;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = (i == n_prompt_tokens - 1);
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "Error: llama_decode failed while processing prompt.\n");
                free(prompt_tokens);
                goto cleanup;
            }
            ++n_past;
        }
        free(prompt_tokens);

        printf("Codsworth: ");
        llama_token next = sample_token(llama_get_logits_ith(ctx, 0), n_vocab, temperature);

        for (int32_t generated = 0; generated < max_gen_tokens; ++generated) {
            if (next == eos) {
                break;
            }

            (void) print_token_piece(vocab, next);

            batch.n_tokens = 1;
            batch.token[0] = next;
            batch.pos[0] = n_past;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = 1;
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "\nError: llama_decode failed while generating.\n");
                goto cleanup;
            }
            ++n_past;

            next = sample_token(llama_get_logits_ith(ctx, 0), n_vocab, temperature);
        }
        printf("\n");
    }

cleanup:
    llama_batch_free(batch);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return 0;
}
