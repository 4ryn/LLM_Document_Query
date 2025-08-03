# save this as uninstall_all.py inside your project directory
import subprocess

def uninstall_all_packages():
    try:
        result = subprocess.run(['pip', 'freeze'], capture_output=True, text=True, check=True)
        packages = result.stdout.strip().split('\n')
        package_names = [pkg.split('==')[0] for pkg in packages if pkg]

        if not package_names:
            print("No packages found to uninstall.")
            return

        print(f"Attempting to uninstall {len(package_names)} packages...")
        subprocess.run(['pip', 'uninstall', '-y'] + package_names, check=True)
        print("All packages uninstalled successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error during uninstallation: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    uninstall_all_packages()