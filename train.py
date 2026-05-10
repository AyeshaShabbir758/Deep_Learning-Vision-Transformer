from src.utils import save_metrics, log_to_csv, save_checkpoint

def main_train_loop():
    # ... setup code ...
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        # --- [TRAINING & VALIDATION STEP] ---
        train_loss, val_loss, val_acc = run_epoch() 

        # --- [INTEGRATION LOGIC: CSV LOGGING] ---
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        }
        log_to_csv(epoch_metrics, "results/training_log.csv")

        # --- [INTEGRATION LOGIC: CHECKPOINTING] ---
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, "checkpoints/best_model.pt")
            
            # Save the latest best metrics to JSON
            best_metrics = {"accuracy": best_acc, "epoch": epoch + 1}
            save_metrics(best_metrics, "results/improved_metrics.json")

if __name__ == "__main__":
    main_train_loop()