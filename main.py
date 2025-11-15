from src.data_cleaning import clean

def main():
    print("=" * 80)
    print("ML Project - World Data Analysis")
    print("=" * 80)

    print("\nAvailable operations:")
    print("1. Clean data")

    choice = input("\nSelect operation (1): ").strip()

    if choice == '1':
        clean.main()
    else:
        print("Invalid choice. Please select 1.")
if __name__ == "__main__":
    main()