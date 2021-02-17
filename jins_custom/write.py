import os

#fr = open(os.path.join(cfg.OUTPUT_DIR, "coco_instances_results.json"))
fr = open(os.path.join("/home/jinsuby/Desktop/PycharmProjects/detectron2/output/3d_reference_thick1/COCO-Detection-faster_rcnn_R_50_FPN_1x-eps3000-thresh0.5", "coco_instances_results.json"))
data = fr.readline()
data = data[1:-1].split("}")
dic={}
for d in data[:-1]:
    print(d)
    image_idx_s = d.find("image_id") + 12
    image_idx_f = d.find("category_id") - 4
    image = d[image_idx_s:image_idx_f]

    category_idx_s = d.find("category_id") + 14
    category_idx_f = d.find("bbox") - 3
    category = d[category_idx_s:category_idx_f]

    bbox_idx_s = d.find("bbox") + 8
    bbox_idx_f = d.find("score") - 4
    bbox = d[bbox_idx_s:bbox_idx_f]

    score_idx_s = d.find("score") + 8
    score = float(d[score_idx_s:])
    # print("image : ", image, ", category : ", category, ", score : ", score, ", bbox : ", bbox)
    if str(image+"-"+category) in dic:
        if score >= dic[image+"-"+category]:
            dic[image + "-" + category] = score
        else:
            pass
    else:
        dic[image + "-" + category] = score

    print(dic)