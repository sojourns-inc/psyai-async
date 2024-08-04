from app import create
import uvicorn

if __name__ == '__main__':
    app_ = create()
    uvicorn.run(app_, host='0.0.0.0', port=8000)
