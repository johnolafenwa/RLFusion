from rlfusion.datasets.ultrachat_sft import UltraChatSFTDataset
from rlfusion.trainers import SFTTrainer


def main():
    train_dataset = UltraChatSFTDataset(train=True, max_samples=20_000, seed=42)
    eval_dataset = UltraChatSFTDataset(train=False, max_samples=2_000, seed=42)

    trainer = SFTTrainer(
        model="Qwen/Qwen3-4B-Base",
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_steps=1_000,
        batch_size=1,
        eval_steps=100,
        saving_steps=100,
        logging_steps=10,
        max_seq_len=2048,
        optimizer_args={"lr": 1e-5},
        output_dir="./ultrachat_sft_qwen3_4b",
    )

    trainer.train()


if __name__ == "__main__":
    main()
