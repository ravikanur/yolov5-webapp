import base64


def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open("./app/predictor_yolo/data/images/" + fileName, 'wb') as f:
        f.write(imgdata)
        f.close()


def encodeImageIntoBase64(croppedImagePath):
    print('before encode')
    with open("./app/predictor_yolo/data/output/" + croppedImagePath, "rb") as f:
        print('after encode')
        return base64.b64encode(f.read())
