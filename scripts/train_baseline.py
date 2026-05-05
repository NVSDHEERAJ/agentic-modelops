from src.training.dataset import FraudDataset
from src.training.model_trainer import ModelTrainer
from src.training.evaluator import ModelEvaluator
from src.training.model_registry import ModelRegistry

def main():
    # Load and prepare data
    dataset = FraudDataset()
    X_train, y_train, X_val, y_val = dataset.get_train_validation_data()

    # Train model
    trainer = ModelTrainer()
    model = trainer.train(X_train, y_train)

    # Evaluate model
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(model, X_val, y_val)

    # Save model and metadata
    registry = ModelRegistry()
    registry.save_model(model)
    registry.save_metadata(features = X_train.columns.tolist(), metrics = metrics)
    registry.save_metrics(metrics)

    print("Baseline model trained and saved with metrics:")
    print(metrics)

if __name__ == "__main__":
    main()