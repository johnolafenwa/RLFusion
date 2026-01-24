from rlfusion.datasets.capybara_sft import CapyBaraSFTDataset
from rlfusion.trainers import SFTTrainer


def main():

    train_dataset = CapyBaraSFTDataset(
        train=True
    )

    eval_dataset = CapyBaraSFTDataset(
        train=False
    )

    trainer = SFTTrainer(
        model="Qwen/Qwen3-4B-Base",
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_steps=300,
        eval_steps=50,
        saving_steps=50,
        logging_steps=10,
        enable_wandb=True,
        output_dir="./capybara_sft_qwen3_4b",
        max_seq_len=2048,
        optimizer_args={"lr": 1e-5}
    )
    
    trainer.train()

if __name__ == "__main__":

    main()

