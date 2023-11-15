import requests
from pathlib import Path
from random_user_agent.user_agent import UserAgent
from random_user_agent.params import HardwareType, SoftwareName, OperatingSystem, Popularity

def download_data(data_dir="../data/"):
    """
    Download the freely available CSVs from Yahoo Finance for Pfizer.

    Args:
        data_dir (string): Relative file path to the directory to download data.
    """
    # Make data_dir a Path object for convenience
    data_dir = Path(data_dir)

    # Tuples of files to download
    data_files = (
        ("historical.csv", "https://query1.finance.yahoo.com/v7/finance/download/PFE?period1=76204800&period2=1698105600&interval=1d&events=history&includeAdjustedClose=true"),
        ("dividends.csv", "https://query1.finance.yahoo.com/v7/finance/download/PFE?period1=76204800&period2=1698105600&interval=1d&events=div&includeAdjustedClose=true"),
        ("splits.csv", "https://query1.finance.yahoo.com/v7/finance/download/PFE?period1=76204800&period2=1698105600&interval=1d&events=split&includeAdjustedClose=true"),
    )

    # Setup a generator of random User Agents
    user_agent_rotator = UserAgent(
            software_names=[SoftwareName.CHROME.value, SoftwareName.FIREFOX.value],
            operating_systems=[OperatingSystem.LINUX.value, OperatingSystem.WINDOWS.value],
            hardware_types=[HardwareType.COMPUTER.value],
            popularity=[Popularity.POPULAR.value],
    )

    # Make sure all files are downloaded
    data_dir.mkdir(parents=True, exist_ok=True)
    for data_file in data_files:
        if not (data_path := data_dir / data_file[0]).exists():
            # Notify user that we're downloading a file
            print(f"Downloading the {data_file[0]}.... ", end="")

            # User Agents required to trick Yahoo filter
            headers = {"User-Agent": user_agent_rotator.get_random_user_agent()}
            resp = requests.get(data_file[1], headers=headers)
            
            # Check for errors while retrieving file
            resp.raise_for_status()

            # Save file in data_dir
            with open(data_path, "wb") as f:
                f.write(resp.content)

            # Notify user that we're done download a file
            print("Done!")


if __name__ == "__main__":
    download_data()
