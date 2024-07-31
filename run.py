import os
import uvicorn

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host='0.0.0.0', port=port, reload=True)
