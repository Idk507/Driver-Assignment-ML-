import pandas as pd
import logging
from src.utils.store import AssignmentStore

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    store = AssignmentStore()
    train_data = store.get_submission("train_data.csv")

    # Target distribution
    target_counts = train_data['target'].value_counts(normalize=True)
    logger.info(f"Target distribution: {target_counts.to_dict()}")

    # Feature correlations
    features = ['driver_distance', 'event_hour', 'historical_completed_bookings', 'driver_acceptance_rate', 'wait_time']
    available_features = [f for f in features if f in train_data.columns]
    if not available_features:
        logger.warning("No features available for correlation analysis")
        return
    correlations = train_data[available_features + ['target']].corr()
    logger.info(f"Feature correlations:\n{correlations}")

    # Feature statistics
    stats = train_data[available_features].describe()
    logger.info(f"Feature statistics:\n{stats}")

if __name__ == "__main__":
    main()