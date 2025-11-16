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

    choice = input("\nSelect operation (1-3): ").strip()

    if choice == '1':
        clean.main()
    elif choice == '2':
        linear_regression.main()
    elif choice == '3':
        linear_regression.main(use_l1=True, alpha=1.0)
    else:
        print("Invalid choice. Please select 1-3.")

if __name__ == "__main__":
    main()