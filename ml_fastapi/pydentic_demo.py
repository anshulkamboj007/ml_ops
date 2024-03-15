from pydantic import BaseModel
from typing import List

data = {
    "name": "Murthy",
    "age": "28",
    "course": "MLOps BootCamp",
    "ratings": [4, 4, "4", "5", 4]
}


class Instructor(BaseModel):
    name: str
    age: int
    course: str
    ratings: List[int] = []


user = Instructor(**data)

print(f"Found a Instructor: {user}")