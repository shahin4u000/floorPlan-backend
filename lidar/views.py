from django.shortcuts import render
from django.http import HttpResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.exceptions import ParseError
from rest_framework.parsers import FileUploadParser
from lidar.create_plot import floorPlanPlotting
from lidar.data_pre_process import *
###########



# Create your views here.


class lidarApi(APIView):
    def post(self, request):
        width = request.data["width"]
        print(width)
        lidarData= request.data["data"]
           
        result, sects = parseData(lidarData)    
        plot = floorPlanPlotting(result, width, sects)
        #check_request = request.data
        #plot = plotting(check_request)

        return Response(plot)



