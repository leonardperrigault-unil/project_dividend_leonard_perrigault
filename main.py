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

    choice = input("\nSelect operation (1-2): ").strip()

    if choice == '1':
        clean.main()
    elif choice == '2':
        linear_regression.main()
    else:
        print("Invalid choice. Please select 1 or 2.")

if __name__ == "__main__":
    main()