from skimage import io
from sklearn.svm import SVC
from sklearn.externals import joblib

from src import ORB
from src.BRIEFbinaryDescription import BRIEF_skimag

if __name__ == '__main__':

    images = io.imread_collection("images/big/*.jpg")
    learnNames = []
    learnKeyPoints = []

    for image, file in zip(images, images.files):
        #cv2 ORB version
        # keyPoints = ORB.ORB(image)
        # x = []
        # for point in keyPoints:
        #     x.append(point.pt[0])
        #     x.append(point.pt[1])
        # if len(x) > 300:
        #     learnNames.append(file[11])
        #     learnKeyPoints.append(x)

        #skimage BRIEF version
        keyPoints = BRIEF_skimag(image)
        if len(keyPoints)>0:
            x = []
            for point in keyPoints:
                x.append(point[0])
                x.append(point[1])
            if len(x) > 4:
                learnNames.append(file[11])
                learnKeyPoints.append(x)

    minSize = len(learnKeyPoints[0])
    for points in learnKeyPoints:
        lenght = len(points)
        if minSize > lenght:
            minSize = lenght

    newLearnKeyPoints = []
    for points in learnKeyPoints:
        newLearnKeyPoints.append(points[:minSize])

    clf = SVC()

    clf.fit(newLearnKeyPoints, learnNames)
    with open('BriefDataSet.pkl', "wb") as file:
        joblib.dump(clf, file)
