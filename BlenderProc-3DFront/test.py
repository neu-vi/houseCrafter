def findMedianSortedArrays(nums1, nums2):
    """
    :type nums1: List[int]
    :type nums2: List[int]
    :rtype: float
    """
    start1 = 0
    start2 = 0
    end1 = len(nums1) - 1
    end2 = len(nums2) - 1

    for _ in range((end1+end2+3)//2):
        if start1 > len(nums1) - 1:
            small = nums2[start2]
            start2 += 1
        elif start2 > len(nums2) - 1:
            small = nums1[start1]
            start1 += 1
        elif nums1[start1] > nums2[start2]:
            small = nums2[start2]
            start2 += 1
        else:
            small = nums1[start1]
            start1 += 1

        if end1 < 0:
            big = nums2[end2]
            end2 -= 1
        elif end2 < 0:
            big = nums1[end1]
            end1 -= 1
        elif nums1[end1] < nums2[end2]:
            big = nums2[end2]
            end2 -= 1
        else:
            big = nums1[end1]
            end1 -= 1
        print(small, big)
    return (big+small)/2
    
nums1 = [1,2]
nums2 = [3,4]
print(findMedianSortedArrays(nums1, nums2))