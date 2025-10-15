import unittest
import torch
from .model import MyTransformer

class TestMyTransformer(unittest.TestCase):

    def test_generate_cache_vs_no_cache_consistency(self):
        """
        Tests that model.generate() with and without caching produce the exact
        same output for a given seed and input.
        """
        # 1. Setup a small model for fast testing
        device = 'cpu'
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        model_config = {
            'vocab_size': 23,
            'd_model': 32,
            'context_size': 64,
            'num_attn_layers': 2,
            'num_query_heads': 4,
            'num_key_value_groups': 2,
            'expansion_factor': 2,
            'dropout_rate': 0.0,
        }
        model = MyTransformer(**model_config).to(device)
        model.eval() # Set to evaluation mode to disable dropout

        # 2. Create a random input context
        batch_size = 2
        context_length = 32
        context = torch.randint(0, model_config['vocab_size'], (batch_size, context_length), device=device)

        # 3. Define generation parameters
        gen_params = {
            'max_new_tokens': 32,
            'top_k': 5,
            'top_p': 0.9,
            'temperature': 1.0
        }

        # 4. Generate text without caching
        torch.manual_seed(1337)
        generated_tokens_no_cache, log_probs_no_cache = model.generate(context, use_cache=False, **gen_params)

        # 5. Generate text with caching
        torch.manual_seed(1337) # Reset seed to ensure identical sampling
        generated_tokens_with_cache, log_probs_cache = model.generate(context, use_cache=True, **gen_params)

        # 6. Assert that the outputs are identical
        self.assertTrue(
            torch.equal(generated_tokens_no_cache, generated_tokens_with_cache),
            "Outputs from generate(use_cache=False) and generate(use_cache=True) should be identical."
        )
        print("\n[TEST PASSED] `generate()` with and without cache produce identical outputs.")

    def test_score_cache_vs_no_cache_consistency(self):
        """
        Tests that model.score() with and without caching produce the same log probabilities.
        """
        # 1. Setup a small model for fast testing
        device = 'cpu'
        model_config = {
            'vocab_size': 23,
            'd_model': 32,
            'context_size': 64,
            'num_attn_layers': 2,
            'num_query_heads': 4,
            'num_key_value_groups': 2,
            'expansion_factor': 2,
            'dropout_rate': 0.0,
        }
        model = MyTransformer(**model_config).to(device)
        model.eval()

        # 2. Create a random input context and a completion to score
        batch_size = 2
        context_length = 16
        completion_length = 16
        context = torch.randint(0, model_config['vocab_size'], (batch_size, context_length), device=device)
        completion = torch.randint(0, model_config['vocab_size'], (batch_size, completion_length), device=device)

        # 3. Score the completion without using the KV cache
        _, log_probs_no_cache = model.score(
            prompt=context,
            completion=completion,
            use_cache=False
        )

        # 4. Score the completion using the KV cache
        _, log_probs_with_cache = model.score(
            prompt=context,
            completion=completion,
            use_cache=True
        )

        # 5. Assert that the log probabilities are identical
        self.assertTrue(
            torch.allclose(log_probs_no_cache, log_probs_with_cache, atol=1e-6),
            "Log probabilities from scoring with and without cache should be identical."
        )
        print("[TEST PASSED] `score()` with and without cache produce identical log probabilities.")

    def test_generate_stops_at_stop_token(self):
        """
        Tests that model.generate() stops when the stop_token is generated.
        """
        # 1. Setup a small model
        device = 'cpu'
        model_config = {
            'vocab_size': 50,
            'd_model': 32,
            'context_size': 64,
            'num_attn_layers': 1,
            'num_query_heads': 2,
            'num_key_value_groups': 1,
            'dropout_rate': 0.0,
        }
        model = MyTransformer(**model_config).to(device)
        model.eval()

        # 2. Define parameters
        batch_size = 1
        context_length = 10
        stop_token = 42
        context = torch.randint(0, model_config['vocab_size'], (batch_size, context_length), device=device)

        # 3. Force the model to output the stop_token by overriding a weight
        # This makes the test deterministic and independent of sampling randomness
        with torch.no_grad():
            # Set the bias for the stop_token to a very high value
            model.W_o.bias[stop_token] = 1e9

        # 4. Generate text
        generated_tokens, _ = model.generate(
            context,
            max_new_tokens=20,
            stop_token=stop_token,
            temperature=0.001 # Low temperature to make it deterministic
        )

        # 5. Assert that generation stopped right after the stop_token
        self.assertEqual(generated_tokens.shape[1], context_length + 1, "Generation should stop immediately after the stop token.")
        self.assertEqual(generated_tokens[0, -1].item(), stop_token, "The last generated token should be the stop token.")
        print("[TEST PASSED] `generate()` stops correctly at the specified stop_token.")

if __name__ == '__main__':
    unittest.main()
