from src.data_cleaning import clean
from src.models import linear_regression
from src.config import SEPARATOR_WIDTH

def main():
    print("=" * SEPARATOR_WIDTH)
    print("ML Project - World Data Analysis")
    print("=" * SEPARATOR_WIDTH)

    print("\nAvailable operations:")
    print("1. Clean data")
    print("2. Train Linear Regression")
    print("3. Train Lasso (L1)")
    print("4. Train Ridge (L2)")

    choice = input("\nSelect operation (1-4): ").strip()

    if choice == '1':
        clean.main()
    elif choice == '2':
        linear_regression.main(regularization="none")
    elif choice == '3':
        linear_regression.main(regularization="l1", alpha=1.0)
    elif choice == '4':
        linear_regression.main(regularization="l2", alpha=1.0)
    else:
        print("Invalid choice. Please select 1-4.")

if __name__ == "__main__":
    main()