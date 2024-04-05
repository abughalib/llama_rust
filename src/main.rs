use std::io::Write;

use llama_cpp::standard_sampler::StandardSampler;
use llama_cpp::{LlamaModel, LlamaParams, SessionParams};

fn main() {
    let params = LlamaParams {
        n_gpu_layers: 100000,
        split_mode: llama_cpp::SplitMode::Layer,
        main_gpu: 0,
        vocab_only: false,
        use_mmap: false,
        use_mlock: false,
    };

    let model = LlamaModel::load_from_file("D:\\AI\\models\\phi-2.Q4_0.gguf", params)
        .expect("Could not load model");

    let mut ctx = model
        .create_session(SessionParams::default())
        .expect("Failed to create session");

    ctx.advance_context("The story about the best planet is the universe ")
        .unwrap();

    let max_tokens = 8196;
    let mut decoded_tokens = 0;

    let completions = ctx
        .start_completing_with(StandardSampler::default(), 1024)
        .into_strings();

    for completion in completions {
        print!("{completion}");
        let _ = std::io::stdout().flush();

        decoded_tokens += 1;

        if decoded_tokens > max_tokens {
            break;
        }
    }
}
