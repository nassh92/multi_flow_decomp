import numpy as np
import matplotlib.pyplot as plt 


def intersect_common_end_point (p1, p2, p3, p4):
    # In the case where the two segments share an endpoint, return True if the cos of the angle between them is 0 and False otherwise
    # In the case where the two segments do not share an endpoint, return None
    if p1 == p3:
        pp2 = (p2[0] - p1[0], p2[1] - p1[1])
        pp4 = (p4[0] - p1[0], p4[1] - p1[1]) 
        cos = np.dot(pp2, pp4)/(np.linalg.norm(pp2) * np.linalg.norm(pp4))
        return not bool(cos)
    elif p1 == p4:
        pp2 = (p2[0] - p1[0], p2[1] - p1[1])
        pp3 = (p3[0] - p1[0], p3[1] - p1[1]) 
        cos = np.dot(pp2, pp3)/(np.linalg.norm(pp2) * np.linalg.norm(pp3))
        return not bool(cos)
    elif p2 == p3:
        pp1 = (p1[0] - p2[0], p1[1] - p2[1])
        pp4 = (p4[0] - p2[0], p4[1] - p2[1]) 
        cos = np.dot(pp1, pp4)/(np.linalg.norm(pp1) * np.linalg.norm(pp4))
        return not bool(cos)
    elif p2 == p4:
        pp1 = (p1[0] - p2[0], p1[1] - p2[1])
        pp3 = (p3[0] - p2[0], p3[1] - p2[1]) 
        cos = np.dot(pp1, pp3)/(np.linalg.norm(pp1) * np.linalg.norm(pp3))
        return not bool(cos)
    return None


def intersect_segments(p1, p2, p3, p4):
    """if p1 == p3 or p1 == p4 or p2 == p3 or p2 == p4:
        return False"""
    # Treating the case where [P1, P2] and [P3, P4] share an endpoint
    intrs = intersect_common_end_point (p1, p2, p3, p4)
    if intrs is not None:
        return intrs
    
    # Return false if segments [P1, P2] and [P3, P4] are both parallel to the ordinate axis
    if p1[0] == p2[0] and p3[0] == p4[0]:
        return False
    
    # Return false if [P_1, P_2] and [P_3, P_4] are parallel, else judge by the intersection point
    if p1[0] == p2[0]: # Case [P1, P2] is parallel to the ordinate axis, and [P3, P4] is not
        a = (p4[1] - p3[1]) / (p4[0] - p3[0])
        b = p3[1] - a * p3[0]
        x_intrs = p1[0]
        y_intrs = a * x_intrs + b
    
    elif p3[0] == p4[0]: # Case [P3, P4] is parallel to the ordinate axis, and [P1, P2] is not
        a = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b = p1[1] - a * p1[0]
        x_intrs = p3[0]
        y_intrs = a * x_intrs + b

    else: # Case where [P1, P2] and [P3, P4] are not parallel to the ordinate axis
        a1 = (p2[1] - p1[1]) / (p2[0] - p1[0])
        a2 = (p4[1] - p3[1]) / (p4[0] - p3[0])
        if a1 == a2: # Return False if [P1, P2] and [P3, P4] are parallel
            return False

        # Process intersection point of straight lines (P1, P2) and (P3, P4) if (P1, P2) and (P3, P4) are not parallel
        b1 = p1[1] - a1 * p1[0]
        b2 = p3[1] - a2 * p3[0]
        x_intrs = (b1 - b2) / (a2 - a1)
        y_intrs = a1 * x_intrs + b1

    # Return true if the segments [P1, P2] and [P3, P4] do intersect and false otherwise
    if  min(p1[0], p2[0]) <= x_intrs <= max(p1[0], p2[0]) and \
        min(p1[1], p2[1]) <= y_intrs <= max(p1[1], p2[1]) and \
        min(p3[0], p4[0]) <= x_intrs <= max(p3[0], p4[0]) and \
        min(p3[1], p4[1]) <= y_intrs <= max(p3[1], p4[1]):
        return True
    return False


def intersect(seg, arcs):
    for arc in arcs:
        if intersect_segments(seg[0], seg[1], arc[0], arc[1]): return True
    return False


if __name__ == "__main__":
    test_names = ["general_case_intersect", "general_case_seg_not_intersect",
                  "share_endpoints", "both_parallel_ordinate", 
                  "first_parallel_ordinate", "first_parallel_ordinate_not_intersect", "second_parallel_ordinate",
                  "general_case_parallel"]

    for test_name in test_names:
        if test_name == "general_case_intersect":
            p1, p2 = (150, 50), (550, 350)
            p3, p4 = (50, 350), (450, 150)
        
        elif test_name == "general_case_seg_not_intersect":
            p1, p2 = (150, 50), (550, 350)
            p3, p4 = (50, 350), (300, 250)
        
        elif test_name == "share_endpoints":
            p1, p2 = (150, 50), (550, 350)
            p3, p4 = (550, 350), (450, 150)

        elif test_name == "both_parallel_ordinate":
            p1, p2 = (150, 50), (150, 350)
            p3, p4 = (450, 350), (450, 150)

        elif test_name == "first_parallel_ordinate":
            p1, p2 = (150, 50), (150, 350)
            p3, p4 = (50, 350), (450, 150)

        elif test_name == "first_parallel_ordinate_not_intersect":
            p1, p2 = (150, 50), (150, 200)
            p3, p4 = (50, 350), (450, 150)

        elif test_name == "second_parallel_ordinate":
            p1, p2 = (150, 50), (550, 350)
            p3, p4 = (450, 350), (450, 150)

        elif test_name == "general_case_parallel":
            p1, p2 = (150, 50), (550, 350)
            p3, p4 = (50, 150), (450, 450)

        print("Intersection status of the segments of case "+test_name+" ", intersect_segments(p1, p2, p3, p4))

        plt.figure()
        plt.title("Test "+test_name, fontsize=15)
        plt.xlabel('x-axis',fontsize=15)
        plt.ylabel('y-axis',fontsize=15)

        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], label='first segment')
        plt.plot([p3[0], p4[0]], [p3[1], p4[1]], label='second segment')
        plt.grid()
        plt.legend()

    plt.show()

