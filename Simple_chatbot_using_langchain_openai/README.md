# Create a Virtual Environment

Create a virtual environment using venv:

   ```bash
   python3 -m venv venv
   ```

# Activate the Virtual Environment

Activate the virtual environment:

- On Windows:
    ```bash
    venv\Scripts\activate
    ```

- On macOS/Linux:
    ```bash
    source venv/bin/activate
    ```

# Install Dependencies

- Install the required packages using pip:
    ```bash
    pip install -r requirements.txt
    ```

### Run the Application

- Finally, run the Streamlit application:
    ```bash
    streamlit run app/main.py
    ```

### Notes
- Ensure that all required dependencies are listed in the requirements.txt file.
- If you face any issues, make sure the virtual environment is activated.

### Troubleshooting
- If you encounter any errors related to missing packages or incorrect versions, try updating the requirements.txt file or reinstalling the dependencies.
- Make sure to deactivate the virtual environment after you're done by running deactivate.
