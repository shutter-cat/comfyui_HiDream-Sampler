# --- Loading Function (Handles NF4, FP8, and default BNB) ---
def load_models(model_type):
    # Double-check core classes loaded
    if not hidream_classes_loaded:
        raise ImportError("Cannot load models: HiDream custom pipeline/transformer classes failed to import.")

    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown or incompatible model_type: {model_type}")

    config = MODEL_CONFIGS[model_type]
    model_path = config["path"]
    is_nf4 = config.get("is_nf4", False)
    is_fp8 = config.get("is_fp8", False)
    scheduler_name = config["scheduler_class"]
    shift = config["shift"]
    requires_bnb = config.get("requires_bnb", False)
    requires_gptq_deps = config.get("requires_gptq_deps", False)

    # Check dependencies again before attempting load
    if requires_bnb and not bnb_available:
         raise ImportError(f"Model type '{model_type}' requires BitsAndBytes but it's not installed.")
    if requires_gptq_deps and (not optimum_available or not autogptq_available):
         raise ImportError(f"Model type '{model_type}' requires Optimum & AutoGPTQ but they are not installed.")

    print(f"--- Loading Model Type: {model_type} ---")
    print(f"Model Path: {model_path}")
    print(f"NF4: {is_nf4}, FP8: {is_fp8}, Requires BNB: {requires_bnb}, Requires GPTQ deps: {requires_gptq_deps}")
    start_mem = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    print(f"(Start VRAM: {start_mem:.2f} MB)")

    # --- 1. Load LLM (Conditional) ---
    # (Keep LLM loading logic the same as before)
    text_encoder_load_kwargs = { # ... (same as before)
        "output_hidden_states": True, "low_cpu_mem_usage": True, "torch_dtype": model_dtype,
    }
    if is_nf4: # ... (same as before)
        llama_model_name = NF4_LLAMA_MODEL_NAME
        print(f"\n[1a] Preparing to load NF4-compatible LLM (GPTQ): {llama_model_name}")
        if accelerate_available: text_encoder_load_kwargs["device_map"] = "auto"; print("     Using device_map='auto' (requires accelerate).")
        else: print("     accelerate not found, will attempt manual CUDA placement.")
    else: # ... (same as before)
        llama_model_name = ORIGINAL_LLAMA_MODEL_NAME
        print(f"\n[1a] Preparing to load Standard LLM (4-bit BNB): {llama_model_name}")
        if bnb_llm_config: text_encoder_load_kwargs["quantization_config"] = bnb_llm_config; print("     Using 4-bit BNB quantization.")
        else: raise ImportError("BNB config required for standard LLM but unavailable.")
        text_encoder_load_kwargs["attn_implementation"] = "flash_attention_2" if hasattr(torch.nn.functional, 'scaled_dot_product_attention') else "eager"

    print(f"[1b] Loading Tokenizer: {llama_model_name}...")
    try: tokenizer = AutoTokenizer.from_pretrained(llama_model_name, use_fast=False)
    except Exception as e: print(f"Error loading tokenizer {llama_model_name}: {e}"); raise
    print("     Tokenizer loaded.")

    print(f"[1c] Loading Text Encoder: {llama_model_name}...")
    print("     (This may take time and download files...)")
    try: text_encoder = LlamaForCausalLM.from_pretrained(llama_model_name, **text_encoder_load_kwargs)
    except Exception as e: print(f"Error loading text encoder {llama_model_name}: {e}"); raise # (Add hints as before)

    if "device_map" not in text_encoder_load_kwargs: # (Same as before)
        print("     Moving text encoder to CUDA..."); try: text_encoder.to("cuda")
        except Exception as e: print(f"     Error moving text encoder to CUDA: {e}. Check model state."); raise

    step1_mem = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    print(f"✅ Text encoder loaded! (VRAM: {step1_mem:.2f} MB)")


    # --- 2. Load Transformer (Conditional) ---
    # (Keep Transformer loading logic the same as before)
    print(f"\n[2] Preparing to load Diffusion Transformer from: {model_path}")
    transformer_load_kwargs = { # ... (same as before)
        "subfolder": "transformer", "torch_dtype": model_dtype, "low_cpu_mem_usage": True
    }
    if is_nf4: print("     Type: NF4 (Quantization included in model files)")
    elif is_fp8: print("     Type: FP8 (Quantization included in model files)") # transformer_load_kwargs["variant"] = "fp8" # Uncomment if needed
    else: print("     Type: Standard (Applying 4-bit BNB quantization)"); # ... (same BNB check)
        if bnb_transformer_4bit_config: transformer_load_kwargs["quantization_config"] = bnb_transformer_4bit_config
        else: raise ImportError("BNB config required for transformer but unavailable.")

    print("     Loading Transformer model...")
    print("     (This may take time and download files...)")
    try: transformer = HiDreamImageTransformer2DModel.from_pretrained(model_path, **transformer_load_kwargs); print("     Moving Transformer to CUDA..."); transformer.to("cuda")
    except Exception as e: print(f"Error loading/moving transformer {model_path}: {e}"); raise

    step2_mem = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    print(f"✅ Transformer loaded! (VRAM: {step2_mem:.2f} MB)")

    # --- 3. Load Scheduler ---
    # Moved before pipeline loading
    print(f"\n[3] Preparing Scheduler: {scheduler_name}")
    scheduler = get_scheduler_instance(scheduler_name, shift)
    print(f"     Using Scheduler: {scheduler_name}")

    # --- 4. Load Pipeline (Passing Pre-loaded Components) ---
    print(f"\n[4] Loading Pipeline definition from: {model_path}")
    print("     Passing pre-loaded Scheduler, Tokenizer, Text Encoder...")
    try:
        # *** CHANGE HERE: Pass loaded components directly ***
        pipe = HiDreamImagePipeline.from_pretrained(
            model_path, # Load non-model files (configs etc) from here
            # Pass the actual objects loaded earlier
            scheduler=scheduler,
            tokenizer_4=tokenizer,
            text_encoder_4=text_encoder,
            # Still pass None for transformer initially
            transformer=None,
            torch_dtype=model_dtype,
            low_cpu_mem_usage=True
        )
        print("     Pipeline structure loaded, using pre-loaded components.")

    except Exception as e:
        print(f"Error loading pipeline definition {model_path}: {e}")
        # If error persists, it might be unrelated to tokenizer=None
        raise

    # --- 5. Final Pipeline Setup ---
    print("\n[5] Finalizing Pipeline Setup...") # Renumbered step
    print("     Assigning loaded transformer to pipeline...")
    pipe.transformer = transformer # Assign the specific transformer we loaded

    print("     Moving pipeline object to CUDA (final check)...")
    try: pipe.to("cuda")
    except Exception as e: print(f"     Warning: Could not move pipeline object to CUDA: {e}.")

    if is_nf4: # (Same as before)
        print("     Attempting to enable sequential CPU offload for NF4 model...")
        if hasattr(pipe, "enable_sequential_cpu_offload"):
            try: pipe.enable_sequential_cpu_offload(); print("     ✅ Sequential CPU offload enabled.")
            except Exception as e: print(f"     ⚠️ Warning: Failed to enable sequential CPU offload: {e}")
        else: print("     ⚠️ Warning: enable_sequential_cpu_offload() method not found on pipeline.")

    final_mem = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    print(f"✅ Pipeline ready! (VRAM: {final_mem:.2f} MB)")

    return pipe, config
