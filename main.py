import main_task1
import main_task2
import save_mapping
import os


def main():
    if not os.path.exists("mapping.txt"):
        save_mapping.save()
    main_task1.main()
    input("Press enter to continue with Task 2:")
    main_task2.main()

if __name__ == '__main__':
    main()