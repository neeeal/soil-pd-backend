from pymysql import connect
import os
from pymysql.cursors import DictCursor
import dotenv

dotenv.load_dotenv()

# Connect to the database
db = connect(host=os.getenv("DATABASE_URL"),
                    user=os.getenv("USER"),
                    password=os.getenv("PASSWORD"),
                    database=os.getenv("DATABASE_NAME"),
                    cursorclass=DictCursor,
                    # port=int(os.getenv("DATABASE_PORT"))
                    )