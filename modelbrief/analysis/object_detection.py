from ..metrics.object_detection import intersection_over_union
def analyse(true_boxes,pred_boxes):
 vals=[intersection_over_union(a,b) for a,b in zip(true_boxes,pred_boxes)]; return {"mean_iou":sum(vals)/len(vals) if vals else 0.0,"matched_boxes":len(vals)}
