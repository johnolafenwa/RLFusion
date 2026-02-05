from rlfusion.datasets.ultrachat_sft import UltraChatSFTDataset
from rlfusion.trainers import SFTTrainer


def main():
    train_dataset = UltraChatSFTDataset(train=True, max_samples=10_000, seed=42)
    eval_dataset = UltraChatSFTDataset(train=False, max_samples=1_000, seed=42)

    trainer = SFTTrainer(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_steps=500,
        eval_steps=50,
        saving_steps=50,
        logging_steps=10,
        max_seq_len=2048,
        optimizer_args={"lr": 2e-5},
        output_dir="./ultrachat_sft_qwen2_5_0_5b",
    )

    trainer.train()


if __name__ == "__main__":
    main()
