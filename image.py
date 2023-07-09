import layoutparser as lp
import cv2
import matplotlib.pyplot as plt
import pdf2image
import numpy as np
import json

pdf_file = '/home/ravi/Desktop/Layout Parser/mergedone1.pdf' # Replace with the path to your PDF file  pdffile name:TB10_MultiColumnTable.pdf
output_dir = '/home/ravi/Desktop/Layout Parser/cropped/' # Replace with the path to your output directory
json_file = '/home/ravi/Desktop/Layout Parser/cropped/output.json' # Replace with the path to your JSON file

class val:
    def __init__(self, imgName, x1 , x2 , y1 , y2):
        self.imgName = imgName
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        
def func() :
    # creating list
    list = []
    jsonList = []
    myset = set()

    model = lp.Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config', 
                                     extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.2],
                                     label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})
        # Load the deep layout model from the layoutparser API 
        # For all the supported model, please check the Model 
        # Zoo Page: https://layout-parser.readthedocs.io/en/latest/notes/modelzoo.html

    images = pdf2image.convert_from_path(pdf_file)
    print(len(images))
    for k, image in enumerate(images):
        
        print('k : ' , k)
        layout = model.detect(image)
        lp.draw_box(image, layout,  box_width=5, box_alpha=0.2, show_element_type=True)
            # Detect the layout of the input image
        print(layout)
        lp.draw_box(image, layout, box_width=3)
#         lp.draw_box(image, layout, box_width=10)
        
            # Show the detected layout of the input image

        table_blocks = lp.Layout([b for b in layout if b.type=='Table'])
        print('table_____blockssss',table_blocks)


        y=1             ## table number on the ith pg


        if table_blocks:
            for blocks in table_blocks:
                plt.imshow(image)
                print("Shape of the actual image", image)
                open_cv_image = np.array(image)
                # Crop image around the detected layout
                segment_image = (blocks.crop_image(open_cv_image))
                # save image
                path = output_dir+'/page_'+str(k+1)+'table_'+str(y)+'.jpg'
                status = cv2.imwrite(path,segment_image)
                print("Image written to file-system : ",status)
                list.append(val('table_'+str(k+1)+'_'+str(y)+":", blocks.block.x_1 ,blocks.block.x_2,blocks.block.y_1,blocks.block.y_2))


                y+=1

            for obj in list:
                print(obj.imgName,obj.x1,obj.x2,obj.y1,obj.y2, sep=' ')
                myset.add(obj);

            plt.imshow(segment_image)

            Pdf = {
                  "path": path,
                  "x1": list[k-1].x1,
                  "x2": list[k-1].x2,
                  "y1": list[k-1].y1,
                  "y2": list[k-1].y2
                }


            k+=1
        else:
            print("No tables were detected on page {}".format(k+1))

    finalJson = []
    for val1 in myset:
    #     print("myyyyyset: " , val.imgName , val.x1)
        finalJson.append({
            "page":val1.imgName[6:7],
            "tableNumber" : val1.imgName[8:9],
            "tableData" : {
                        "x1" : val1.x1,
                        "x2" : val1.x2,
                        "y1" : val1.y1,
                        "y2" : val1.y2
                        }

        })
    print("finaallllJSONNN : " , finalJson)
    json_string = json.dumps(finalJson)
  
    with open(json_file, 'w') as f:
        f.write(json_string)
    data = json.loads(json_string)

    # Loop through each table in the data
    for table in data:
        # Extract the page, table number, and position information
        page = table['page']
        table_number = table['tableNumber']
        x1 = table['tableData']['x1']
        x2 = table['tableData']['x2']
        y1 = table['tableData']['y1']
        y2 = table['tableData']['y2']

        # Generate a unique ID for the table
        table_id = f"page_{page}_table_{table_number}"

        # Generate CSS code to position the table
        css = f"""
        #{table_id} {{
            position: absolute;
            left: {x1}px;
            top: {y1}px;
            width: {x2 - x1}px;
            height: {y2 - y1}px;
        }}
        """

        # Print the CSS code
        print(css)
    return finalJson

func()
