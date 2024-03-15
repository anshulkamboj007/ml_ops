from fastapi import FastAPI

app =FastAPI()
@app.get("/")
async def root():
    return {"message":"hellow world from fastapi"}