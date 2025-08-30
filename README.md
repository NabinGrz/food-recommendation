# AI Coursework: Food Recommendation System

## Dataset

**Note:** The original dataset has been removed from this repository due to size and copyright restrictions.

You can download the dataset from Kaggle:
**[Food.com Recipes and User Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)**

After downloading, place the relevant files in the `data/` directory as required by the scripts.

**Processed data files are not included in this repository.** You must generate them by running the provided data processing scripts after placing the raw dataset in the `data/` directory.

## Repository Structure

- `apps/`: Application scripts (e.g., search app)
- `src/`: Source code for recommendation models and utilities
- `data/`: (Not included) Place raw dataset files here
- `processed_data/`: (Not included) This folder will be populated with processed data files after you run the data processing scripts.
- `requirements.txt`: Python dependencies

## Getting Started

1. Download the dataset from the Kaggle link above.
2. Place the dataset files (e.g., `RAW_recipes.csv`, `RAW_interactions.csv`) in the `data/` directory.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. **Preprocess the data:**
   Run the following command to generate the processed data files:
   ```bash
   python src/process_data_efficient.py
   ```
   This will create the necessary files in the `processed_data/` directory.
5. Run the main application or analysis scripts as needed.

## License

This project is for educational purposes only.
