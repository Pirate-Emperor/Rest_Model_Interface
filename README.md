# Rest_Model_Interface

Developed by Pirate-Emperor, Rest_Model_Interface is a simple web application that provides seat forecasting using a machine learning (ML) model. It is built using Flask, a lightweight web application framework in Python.

## Features

- **Seat Forecasting**: Generates seat forecasting based on user input and preferences.
- **Cluster Identification**: Generating clusters of restaurats based on dishes and customers based on frequency, recency, cost, and head count.
- **Hybrid Model**: Use of both clustering and forecasting to predict reserve seats based on MVP customer cluster
- **ML Model Integration**: Utilizes a pre-trained ML model for predicting user preferences.
- **User Interface**: Offers a clean and easy-to-use web interface for users to interact with the application.
- **Search Functionality**: Allows users to search for movies based on keywords or titles.
- **User Authentication**: Provides a secure user authentication system for personalized recommendations.

## Prerequisites

To run the project, you will need:

- Python 3.x
- Required Python libraries (Flask, numpy, pandas, scikit-learn, etc.)

## Installation

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/Pirate-Emperor/Rest_Model_Interface.git
cd Rest_Model_Interface
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

Run the Flask app:

```bash
python app.py
```

The web application will be running on `http://127.0.0.1:5000/`.

Visit the URL in your web browser and start exploring the seat forecasting.

## Data Source

The project uses a dataset of movie information and user ratings to train the ML model.

## Development

To enhance the project, you can modify the Python scripts and HTML templates in the `src` and `templates` directories, respectively. Some potential areas for improvement include:

- Improving the accuracy of the ML model for better recommendations.
- Enhancing the user interface design for a more engaging experience.
- Incorporating additional features, such as movie trailers or reviews.
- Scaling the application for handling a larger number of users and data.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
