# US-Open-prediction-2026

## Disclaimer
This model was fully developed after the tournamnent was finished. All work during the US Open 2026 was a rough version of the model. Any refinements, improvements and additional features adopted into this repository were made after the tournamnent concluded and did not benefit from or influence live tournamnent outcomes.  

## Project Overview
This project is a machine learning model built to predict the winner of each match for the men's and women's singles for this year's tournamnent. Using data from (https://stats.tennismylife.org/tennis-match-database) I downloaded ATP Tour and WTA Tour dataset to train my model in order to make a prediction for each match going on in the tournmanent. To predict every match this model relies on past history data for each player to generate predictions, comparing prior results between competitors to estimate the probability of the likley winner of a given matchup. 

Challenger Tour data was intentionally excluded as it wasn't necessary for predictions involving ATP and WTA tour matches. As a result, this model was unable to generate predictions for a small number of matches involving players making their debut in the ATP/WTA tour. Due no prior match history this tour was available for them. 

## Installation 
### Prerequisites
- Python 3.12.8 (or higher)
- git
- pip

### Clone this repository 
``` bash
git clone https://github.com/JRob500/US-Open-prediction-2026.git
```

### Go to your terminal
#### Create and activate a virtual environment

**macOS/Linux:**
```bash
python3 -m venv .venv
source venv/bin/activate
```

**Windows**
```bash
python -m venv .venv
source venv/bin/activate 
```

### Install dependencies 

``` bash
cd US-Open-prediction-2026
```
**Windows/Linux**:
```bash
pip install -r requirements.txt 
```
**macOS**:
```bash
pip3 install -r requirements.txt 
```


