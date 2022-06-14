import cv2
import numpy as np
count=0

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def draw_prediction(img, class_id, confidence, x, y, w, h):
    global frame
    global axle_detected
    color = COLORS[class_id]
    label = str(classes[class_id])
    print(label,confidence)
    cv2.rectangle(frame, (x,y), (x+w,y+h), color, 1)
    cv2.putText(frame, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

with open("yolov3-tiny-obj.names", 'r') as f:
    classes = [line.strip() for line in f.readlines()]
    #print(len(classes))

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
net = cv2.dnn.readNet("yolov3-tiny-obj_1000.weights", "yolov3-tiny-obj.cfg")
    
for i in range(1):
    frame = cv2.imread("test4.jpg")                 #Input Image
    if frame is None:
        continue
    axle_detected = False
    scale = 0.003
    Width = frame.shape[1]
    Height = frame.shape[0]
    blob = cv2.dnn.blobFromImage(frame, scale, (416,416), (0,0,0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.5
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
                count=count+1
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    #print(indices)
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(frame, class_ids[i], confidences[i], round(x), round(y), round(w), round(h))
    
    print("No. of Helmet = ", count)
    cv2.imshow("Helmet",frame)
    cv2.waitKey(0)





