import sys
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from mandarin_grader.data.aishell_tar_source import AISHELL3TarDataSource
from mandarin_grader.data.autoregressive_dataset import SyntheticSentenceInfo
from mandarin_grader.data.mel_augmentation import get_preset_config
from scripts.train_v6 import CollateFn

def profile_dataloader(num_workers, augment_preset, log_file):
    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"\n--- Profiling DataLoader (workers={num_workers}, config={augment_preset}) ---")
    data_dir = Path("datasets/aishell3_tar")
    if not data_dir.exists():
        log("Error: datasets/aishell3_tar not found.")
        return

    source = AISHELL3TarDataSource()
    
    t0 = time.time()
    raw_sentences = source.load(data_dir, split="train", max_sentences=2000)
    mel_cache = source.get_mel_cache()
    t1 = time.time()
    
    log(f"Loaded {len(raw_sentences)} sentences and {len(mel_cache)} mels from source in {t1 - t0:.2f}s")
    
    sentences = []
    for s in raw_sentences:
        sentences.append(SyntheticSentenceInfo(
            id=s.id,
            audio_path=s.audio_path,
            text=s.text,
            syllables=s.syllables,
            syllable_boundaries=s.syllable_boundaries,
            sample_rate=s.sample_rate,
            total_samples=s.total_samples,
        ))
        
    log("Initializing FullSentenceDataset...")
    from mandarin_grader.data.full_sentence_dataset import FullSentenceDataset
    
    augment = augment_preset != "none"

    dataset = FullSentenceDataset(
        sentences=sentences,
        sample_rate=16000,
        max_duration_s=2.0,
        max_syllable_position=4,
        augment=augment,
    )
    dataset._mel_cache.update(mel_cache)

    collate_fn = CollateFn(max_frames=200, random_padding=True, augment=augment)
    
    loader = DataLoader(
        dataset, 
        batch_size=128, 
        shuffle=True,
        collate_fn=collate_fn, 
        num_workers=num_workers,
        persistent_workers=num_workers > 0
    )
    
    log("Fetching batches...")
    t2 = time.time()
    
    try:
        for i, batch in enumerate(loader):
            if i >= 10:
                break
            # Force evaluation of tensors to ensure augmentations run
            batch["mel"].sum().item()
            log(f"  Processed batch {i+1}/10 (shape: {batch['mel'].shape})")
    except Exception as e:
        log(f"FAILED: {e}")
        return
        
    t3 = time.time()
    log(f"Processed 10 batches in {t3 - t2:.2f}s ({((t3 - t2) / 10) * 1000:.1f}ms per batch)")

if __name__ == "__main__":
    with open("dataloader_results_py.log", "w", encoding="utf-8") as f:
        profile_dataloader(num_workers=0, augment_preset="none", log_file=f)
        profile_dataloader(num_workers=0, augment_preset="mobile", log_file=f)
        # Multiprocessing test
        profile_dataloader(num_workers=4, augment_preset="mobile", log_file=f)
