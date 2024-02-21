import cv2
import numpy as np
import sys

#class stores a persons x and y location in the image
class person_data:
    def __init__(self, x, y):
	    self.x = x
	    self.y = y


def LocateStudents(file_name):
    #create empty list of persons
    person = []
    #read neural net
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    boxes = []
    indexes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Loading image
    img = cv2.imread('PiCameraCaptures/' + file_name + '.jpg')
    img = cv2.resize(img, None, fx=.5, fy=.5)

    #split the image into 4 equally sized images.
    img_list = split_image(img)
    
    #Run object detection algorithm on the three images
    center_x, center_y, class_ids, boxes, indexes = object_detection(img, net, output_layers)
    #Put person location data into person list

    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        person.append(person_data(center_x[i], center_y[i]))
        if i in indexes:
            print(f'{person[i].x}     {person[i].y}     {str(classes[class_ids[i]])}    {str(class_ids[i])}')
            #cv2.rectangle(img, (person[i].x-50, person[i].y-50), (person[i].x+50, person[i].y+50), (255,255,0),2)
            cv2.rectangle(img, (boxes[i][0], boxes[i][1]), ((boxes[i][0]+boxes[i][2]), (boxes[i][1]+boxes[i][3])), (255,255,0), 2)
            cv2.putText(img, str(classes[class_ids[i]]), (person[i].x, person[i].y+70), font, 3, (0,255,0),3)

    cv2.imwrite('YoloOutputImages/'+file_name+'_out.jpg', img)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#This function splits image into 4 equally sized sub images.
# @Author: Brandon Johnson
# @Date: 4/15/2020
def split_image(img):
    height, width, channels = img.shape
    rows = int(height)
    cols = int(width)
    
    #top left sub image
    img_00 = img[0:rows//2 , 0:cols//2]
    #top right sub image
    img_01 = img[0:rows//2 , cols//2:cols]
    #top  left sub image
    img_10 = img[rows//2:rows , 0:cols//2]
    #top left sub image
    img_11 = img[rows//2:rows , cols//2:cols]
    #list of subimages
    img_list = [img_00, img_01, img_10, img_11]    
    return img_list


#This function runs object detection algorithm 
#and returns two lists, center x and y positions of each object.
# @Author: Brandon Johnson
# @Date: 4/16/2020
def object_detection(img, net, output_layers):
    height, width, channels = img.shape
    print(f'{height}    {width}')
    #empty lists that store x and y coords
    center_x = []
    center_y = []
    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:
                # Object detecteds
                # X,Y coordinate for each person.
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                center_x.append(int(detection[0] * width))
                center_y.append(int(detection[1] * height))
                x = int(int(detection[0] * width)-w/2)
                y = int(int(detection[1] * height)-h/2)
                boxes.append([x,y,w,h])
                class_ids.append(class_id)
                confidences.append(float(confidence))
    #return x and y
    indexes = cv2.dnn.NMSBoxes(boxes,confidences, 0.5, 0.4)
    return center_x, center_y, class_ids, boxes, indexes


def main():
    if sys.argv[1]:
        return LocateStudents(sys.argv[1])

if __name__ == "__main__":
    main()


'''
1. split image function -- done
2. object detection into a function -- done
3. make the output really nice.  -- done 
4. add check for two objects in the same place 

Figure out how to only detect persons.
'''
