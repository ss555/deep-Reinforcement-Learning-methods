from typing import List,Optional
from collections import defaultdict,Counter,deque
from operator import itemgetter
import numpy as np

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def jump(self, nums: List[int]) -> int:
        self.m=10**5
        def h(nums: List[int],j=0,c=0):
            if j==(len(nums)-1):
                if c<self.m:
                    self.m=c
            elif c>=len(nums):
                c=0
                return
            else:
                if nums[j]>0:
                    for i in range(1,nums[j]+1):
                        h(nums,j+i,c+1)
        h(nums)
        return self.m


# def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
#     c=[]
#     def suma(candidates,rs,c,t):
#         if candidates==[]:
#             return
#         rs=rs+[candidates[0]]
#         if (sum(rs))==t:
#             if rs not in c:
#                 c.append(rs)
#         if candidates[]:
#         suma(candidates[1:],rs,c,t)
#         suma(candidates[1:],[candidates[0]],c,t)

#     suma(candidates,[],c,target)
#     return c


        


root = TreeNode(5,TreeNode(4),TreeNode(6,TreeNode(3),TreeNode(7)))
s=Solution()
grid =[["1","1","1","1","0"],
 ["1","1","0","1","0"],
 ["1","1","0","0","0"],
 ["0","0","0","0","0"]]
# [
#   ["1","1","1","1","0"],
#   ["1","1","0","1","0"],
#   ["1","1","0","0","0"],
#   ["0","0","0","0","0"]
# ]
print(s.jump([2,3,1,1,4]))
# print(s.maxSlidingWindow([1,3,-1,-3,5,3,6,7],3))
# print(s.isValidBST(root))

#TODO ddqn overestimation idea is used in SAC