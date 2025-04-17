import logging
import os,shutil
import os.path
from xmlrpc.client import MAXINT
import vtk
import numpy as np
import slicer
import vtk
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
import scipy.ndimage.morphology as scp
import math
import time
#slicer.util.pip_install('xlsxwriter')
#slicer.util.pip_install('pyvista')
import xlsxwriter
#slicer.util.pip_install('scipy')
from scipy.optimize import minimize
#slicer.util.pip_install('pip install torch==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html')
import torch #slicer.util.pip_install('pip install torch==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html')
#slicer.util.pip_install('openpyxl')
import openpyxl
import HDBrainExtractionTool
#slicer.util.pip_install('SimpleITK')
import SimpleITK as sitk
import sitkUtils
import SurfaceRegistration as sf
import sys
from subprocess import call 
#slicer.util.pip_install("monai==1.1.0")
#slicer.util.pip_install("monai[all]") 
#slicer.util.pip_install('numpy==1.26.1')
#slicer.util.pip_install("monai[einops]")

from monai.networks.layers import Norm
from monai.networks.nets import UNet, UNETR
from monai.inferers import sliding_window_inference
#EnsureChannelFirstd,
from monai.transforms import (EnsureChannelFirstd, Compose,CropForegroundd,
                              LoadImaged,Orientationd,RandCropByPosNegLabeld,ScaleIntensityRanged,SaveImage,Spacingd)


torch.cuda.empty_cache()
no ='03'
#AgeGroup = 'UNC-Pediatric1year_Brain_Atlas'
AgeGroup = 'UNC_Adult_Brain_Atlas'
#AgeGroup= 'UNC_Elderly_Brain_Atlas'
#AgeGroup = 'UNC_Pediatric_Brain_Atlas'

file_path_vessel='E:\\Datasets\\Damar\\00.ITKTubeTK'
file_path_brats='E:\\Datasets\\Tumor (BraTS2021)\\3d'


class BrainBiopsy(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "BrainBiopsy"  # TODO: make this more human readable by adding spaces
        self.parent.categories = ["Examples"]  # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Anonymous (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = "AAA"
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = "AAA"  

#
# BrainBiopsyParameterNode
#

class BrainBiopsyWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/BrainBiopsy.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = BrainBiopsyLogic()

        # Connections
        self.ui.inputSelectorHeadMRI.connect("currentNodeChanged(vtkMRMLNode*)", self.updateGUIFromParameterNode)
        #self.ui.inputSelectorMRAVessel.connect("currentNodeChanged(vtkMRMLNode*)", self.updateGUIFromParameterNode)
        #self.ui.inputSelectorMRIBrain.connect("currentNodeChanged(vtkMRMLNode*)", self.updateGUIFromParameterNode)
                
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        # Buttons
        self.ui.brainExtractButton.connect('clicked(bool)', self.onBeyinCikar)
        self.ui.tumorSegmentButton.connect('clicked(bool)', self.onTumorCikar)
        self.ui.vesselSegmentButton.connect('clicked(bool)', self.onDamarCikar)
        self.ui.MRIriskCalculationButton.connect('clicked(bool)', self.onMRIRiskHesaplaEski)
        #self.ui.MRIriskCalculationButtonEski.connect('clicked(bool)', self.onMRIRiskHesaplaEski)
        #self.ui.FMRIriskCalculationButton.connect('clicked(bool)', self.onFMRIRiskHesapla)
        self.ui.extractUpperShellButton.connect('clicked(bool)', self.onExtractUpperShellButton)
        #self.ui.getAtlasButton.connect('clicked(bool)', self.ongetAtlasButton)
        #self.ui.UploadPointsButton.connect('clicked(bool)', self.onUploadPointsButton)
        #self.ui.registrateAtlasButton.connect('clicked(bool)', self.onregistrateAtlasButton)
        #self.ui.cutUpperShellAtlasButton.connect('clicked(bool)', self.oncutUpperShellAtlasButton)


        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self):
        self.removeObservers()

    def enter(self):
        self.initializeParameterNode()

    def exit(self):
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    def onSceneStartClose(self, caller, event):
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self):
        self.setParameterNode(self.logic.getParameterNode())

    def setParameterNode(self, inputParameterNode):
        if self._parameterNode is not None:
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self.updateGUIFromParameterNode()    

    def updateGUIFromParameterNode(self, caller=None, event=None):
        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return       
                
        self._updatingGUIFromParameterNode = False
    def onBeyinCikar(self):                       
        with slicer.util.tryWithErrorDisplay("Beyin cikarmada hata var.", waitCursor=True):
            self.logic.ProcessBeyinCikar(self.ui.inputSelectorHeadMRI.currentNode())

    def onMRIRiskHesapla(self):                       
        with slicer.util.tryWithErrorDisplay("Risk hesaplamada hata var.", waitCursor=True):
            self.logic.MRAVeinRiskButton()
    def onMRIRiskHesaplaEski(self):                       
        with slicer.util.tryWithErrorDisplay("Risk hesaplamada hata var.", waitCursor=True):
            self.logic.MRAVeinRiskButtonEski()
    def onFMRIRiskHesapla(self):                       
        with slicer.util.tryWithErrorDisplay("FMRI Risk hesabında hata var.", waitCursor=True):
            self.logic.ProcessFMRIRiskHesapla()


    def onTumorCikar(self):                        
        with slicer.util.tryWithErrorDisplay("Tumor cikarmada hata var.", waitCursor=True):
                self.logic.ProcessTumorCikar(self.ui.inputSelectorHeadMRI.currentNode())
                
    def onDamarCikar(self):                        
        with slicer.util.tryWithErrorDisplay("Damar cikarmada hata var.", waitCursor=True):
            self.logic.ProcessDamarCikar(self.ui.inputSelectorHeadMRI.currentNode())


    def onExtractUpperShellButton(self):                        
        with slicer.util.tryWithErrorDisplay("Beyin Üst kabuk cikarmada hata var.", waitCursor=True):
            self.logic.MRABrainTopShellButton()

    def ongetAtlasButton(self):                        
        with slicer.util.tryWithErrorDisplay("Atlas modeli getirmede hata var.", waitCursor=True):       
            self.logic.GetAtlasModelButton()


    def oncutUpperShellAtlasButton(self):                        
        with slicer.util.tryWithErrorDisplay("Atlas modeli kesmede hata var.", waitCursor=True):       
            self.logic.cutUpperShellAtlas()



    def onUploadPointsButton(self):                        
        with slicer.util.tryWithErrorDisplay("atlas ve beyin uç noktalarını bulmada  hata var.", waitCursor=True):       
            self.logic.FindModelEndPointButton()


    def onregistrateAtlasButton(self):                        
        with slicer.util.tryWithErrorDisplay("örtüştürmede hata var.", waitCursor=True):       
            self.logic.RegistrationButton()

#
# BrainBiopsyLogic
#

modulPath=os.path.dirname(os.path.realpath(__file__))
class BrainBiopsyLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)
    
    def ProcessBeyinCikar(self, MRI):
        HDBetLogic = HDBrainExtractionTool.HDBrainExtractionToolLogic()                                        
               
        #MRA = slicer.util.loadVolume('D:\\Datasets\\ITKTubeTK\\Normal-0'+no+'\\MRA\\Normal0'+no+'-MRA.mha'); MRA.SetName('MRA')
        MRI.SetName('MRA')
        # beyin ve maskesinin dugumlerini olustur                          
        MRABrainSeg = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode','MRABrainSeg')
        
        # beyni cikar ve dugumlere ata
        HDBetLogic.process(MRI,0,MRABrainSeg,0)    
        MRABrainSeg.CreateClosedSurfaceRepresentation() 

    def findCenterTumor(self, tumorSegment):
        
        visibleSegmentIds = vtk.vtkStringArray()
        tumorSegment.GetDisplayNode().GetVisibleSegmentIDs(visibleSegmentIds)
        segmentId = visibleSegmentIds.GetValue(0)

        tumorCenterCoordinates = [0,0,0]
        tumorSegment.GetSegmentCenter(segmentId, tumorCenterCoordinates)
        return tumorCenterCoordinates     
    
    def ProcessTumorCikar(self,T1):
        print(modulPath)
        # Create new empty folder
        tempFolder = slicer.util.tempDirectory()
        name=T1.GetName()
        input_file = tempFolder+"/"+name+".nii.gz"
        #output_file = tempFolder+"/hdbet-output.nii.gz"

        ### input normalization
        imageData = T1.GetImageData()
        voxels=vtk.util.numpy_support.vtk_to_numpy(imageData.GetPointData().GetScalars())

        normalizedVoxels=((voxels-np.min(voxels)) /(np.max(voxels)-np.min(voxels)))*255
        imageData.GetPointData().SetScalars(vtk.util.numpy_support.numpy_to_vtk(normalizedVoxels))
        T1.Modified()


        # Write input volume to file
        volumeStorageNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLVolumeArchetypeStorageNode")
        volumeStorageNode.SetFileName(input_file)
        volumeStorageNode.WriteData(T1)
        volumeStorageNode.UnRegister(None)

        slicer.mrmlScene.RemoveNode(T1)

        ## test normalization        
        T1 = slicer.util.loadVolume(input_file); #T1.SetName('T1')




        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

        model_dir=os.path.join(modulPath,'tumorModel')


        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),#her katmandaki convolüsyon katmanları 
            strides=(2, 2, 2, 2),# sıçrama katsayısı
            num_res_units=2,
            norm=Norm.INSTANCE,
        ).to(device)

        model.load_state_dict(torch.load(os.path.join(model_dir, "tumor_model.pth")))
        model.eval()
        model.to(device)

        roi_size = (96, 96, 96)
        pixdim = (1, 1, 1)
        class_size = 2
        #file_path = input_file
        model_path = os.path.join(model_dir, "tumor_model.pth")
        image_path = input_file
        pre_transforms = Compose(
            [
                LoadImaged(keys="image", image_only=True),
                
                EnsureChannelFirstd(keys="image"),
                
                ScaleIntensityRanged(keys=["image"], a_min=-175.0, a_max=250.0, b_min=0.0, b_max=1.0, clip=True),
                
                #CropForegroundd(keys="image", source_key="image"), 
                
                Orientationd(keys="image", axcodes="RAS"),
            
                #Spacingd(keys="image", pixdim=pixdim, mode="bilinear"),         
            ]
        )
        mri_image = pre_transforms({'image':image_path})


        #pip install slicer
        #import slicer

        #ijkToRAS = slicer.util.vtkMatrixFromArray(utils['imageDirections'])

        #volumeNode2 = slicer.util.addVolumeFromArray(utils['np_data'], ijkToRAS=ijkToRAS, nodeClassName="vtkMRMLLabelMapVolumeNode")

        model.load_state_dict(torch.load(model_path))
        model.eval()
        model.to(device)
        print()
        with torch.no_grad():
            
            test_inputs = torch.unsqueeze(mri_image['image'], 1).to(device)
            
            test_outputs = sliding_window_inference(test_inputs, roi_size, 4, model, overlap=0.8)
            saver = SaveImage(model_dir, output_ext=".nii.gz", output_postfix="seg")
            test_outputs = torch.argmax(test_outputs, dim=1).squeeze(0)
            saver(test_outputs)

        ### remove input file
        os.remove(input_file)
        str=os.path.join(name,name+"_seg.nii.gz")
        tumor = slicer.util.loadSegmentation(os.path.join(model_dir,str)); tumor.SetName('tumor')
        tumorLabelMap=slicer.util.loadLabelVolume(os.path.join(model_dir,str)); tumorLabelMap.SetName('tumorLabelMap')
        
        output_file=os.path.join(model_dir,str)
        output_folder=os.path.join(model_dir,name)
        shutil.rmtree(output_folder)
        tumorCenterCoordinates=self.findCenterTumor(tumor)
        # Add Target Markup            
        TargetNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode","Target")        
        n = TargetNode.AddControlPoint(tumorCenterCoordinates[0], tumorCenterCoordinates[1], tumorCenterCoordinates[2])
        TargetNode.SetNthControlPointLabel(n, "Target")

        #os.remove(output_file)
        #os.rmdir(output_folder)

        '''if outputVolume:
            volumeStorageNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLVolumeArchetypeStorageNode")
            volumeStorageNode.SetFileName(output_file)
            volumeStorageNode.ReadData(outputVolume)
            volumeStorageNode.UnRegister(None)
            os.remove(output_file)

        if outputSegmentation:
            segmentationStorageNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLSegmentationStorageNode")
            segmentationStorageNode.SetFileName(output_segmentation_file)
            segmentationStorageNode.ReadData(outputSegmentation)
            segmentationStorageNode.UnRegister(None)
            os.remove(output_segmentation_file)'''


        '''
        #slicer.util.pip_install('einops') 
        with open('C:\\Users\\msahin\\Desktop\\BrainBiopsy\\BrainBiopsy\\BrainBiopsy\\log.txt', 'w') as f:
            call(['python' , 'C:\\Users\\msahin\\Desktop\\BrainBiopsy\\BrainBiopsy\\BrainBiopsy\\test.py'], stdout=f)
        output_directory ="C:\\Users\\msahin\\Desktop\\BrainBiopsy\\BrainBiopsy\\BrainBiopsy\\TumorSonuc"
        slicer.mrmlScene.SetRootDirectory("C:\\Users\\msahin\\Desktop\\BrainBiopsy\\BrainBiopsy\\BrainBiopsy\\TumorSonuc")
        #slicer.util.loadVolume(os.path.join(output_directory,"tumor.nii.gz"),properties={},returnNode=False)
        tumorVolNode=slicer.util.loadVolume("C:\\Users\\msahin\\Desktop\\BrainBiopsy\\BrainBiopsy\\BrainBiopsy\\TumorSonuc\\tumor.nii.gz",{"name": "tumor"})
        tumorSegNode=slicer.util.loadSegmentation("C:\\Users\\msahin\\Desktop\\BrainBiopsy\\BrainBiopsy\\BrainBiopsy\\TumorSonuc\\tumor.nii.gz",{"name": "tumorseg"})
        f = slicer.util.getNode('tumorseg')
        tumorCenter=self.findCenterTumor(f)
        markup_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        tumC=np.array([tumorCenter])
        slicer.util.updateMarkupsControlPointsFromArray(markup_node,tumC)
        markup_node.GetDisplayNode().SetPointLabelsVisibility(False)
        #f.GetNthFiducialPosition(0,tumorCenter)
        print(tumorCenter)'''
        
        '''if not Beyin:
            logging.info('Beyin yüklenemedi.')
        else:                                
            NodeList = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
            ID = NodeList.GetItemByDataNode(Beyin)
            clonedID = slicer.modules.subjecthierarchy.logic().CloneSubjectHierarchyItem(NodeList, ID)
            Tumor = NodeList.GetItemDataNode(clonedID)
            Tumor.SetName("Tumor")
            
            import numpy as np
            voxels = slicer.util.arrayFromVolume(Tumor)
            voxels[:] = voxels + np.random.normal(0.0, 25.0, size=voxels.shape)
            slicer.util.arrayFromVolumeModified(Tumor)          
            slicer.util.setSliceViewerLayers(background=Tumor) 
            logging.info('Tümor bulundu')'''

    def ProcessDamarCikar(self,MRA):

        logging.info('Damar bulma yazılımı başladı...')
        print(modulPath)
        # Create new empty folder
        tempFolder = slicer.util.tempDirectory()
        name=MRA.GetName()
        input_file = tempFolder+"/"+name+".nii.gz"
        #output_file = tempFolder+"/hdbet-output.nii.gz"

        ### input normalization
        imageData = MRA.GetImageData()
        voxels=vtk.util.numpy_support.vtk_to_numpy(imageData.GetPointData().GetScalars())

        normalizedVoxels=((voxels-np.min(voxels)) /(np.max(voxels)-np.min(voxels)))*255
        imageData.GetPointData().SetScalars(vtk.util.numpy_support.numpy_to_vtk(normalizedVoxels))
        MRA.Modified()


        # Write input volume to file
        volumeStorageNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLVolumeArchetypeStorageNode")
        volumeStorageNode.SetFileName(input_file)
        volumeStorageNode.WriteData(MRA)
        volumeStorageNode.UnRegister(None)

        slicer.mrmlScene.RemoveNode(MRA)

        ## test normalization        
        MRA = slicer.util.loadVolume(input_file); #MRA.SetName('MRA')

        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        model_dir=os.path.join(modulPath,'damarModel')
        roi_size = (96,96,96)
        pixdim = (0.5, 0.5, 0.8)

        model = UNETR(
            in_channels=1,
            out_channels=2,
            img_size=roi_size,
            feature_size=16,
            hidden_size=1024,
            mlp_dim=3072,
            num_heads=16,
            pos_embed="conv",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
        ).to(device)

        model.load_state_dict(torch.load(os.path.join(model_dir, "damar_model.pth")))
        model.eval()
        model.to(device)


        class_size = 2
        #file_path = input_file
        model_path = os.path.join(model_dir, "damar_model.pth")
        image_path = input_file
        pre_transforms = Compose(
            [
                LoadImaged(keys="image", image_only=True),
                
                EnsureChannelFirstd(keys="image"),
                
                ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=64.0, b_min=0.0, b_max=1.0, clip=True),
                
                #CropForegroundd(keys="image", source_key="image"), 
                
                Orientationd(keys="image", axcodes="RAS"),
            
                #Spacingd(keys="image", pixdim=pixdim, mode="bilinear"),         
            ]
        )
        mri_image = pre_transforms({'image':image_path})
        mri_image['image'].shape


        #pip install slicer
        #import slicer

        #ijkToRAS = slicer.util.vtkMatrixFromArray(utils['imageDirections'])

        #volumeNode2 = slicer.util.addVolumeFromArray(utils['np_data'], ijkToRAS=ijkToRAS, nodeClassName="vtkMRMLLabelMapVolumeNode")

        model.load_state_dict(torch.load(model_path))
        model.eval()
        model.to(device)
        print()
        with torch.no_grad():
            test_inputs = torch.unsqueeze(mri_image['image'], 1).to(device)
            test_outputs = sliding_window_inference(test_inputs, roi_size, 4, model, overlap=0.8)
            #test_outputs = torch.argmax(test_outputs, dim=1).detach().cpu().numpy()[0, :, :, :]
            test_outputs = torch.argmax(test_outputs, dim=1).squeeze(0)

            saver = SaveImage(model_dir, output_ext=".nii.gz", output_postfix="seg")
            saver(test_outputs)

        
         # 2. yöntem

        ### remove input file
        os.remove(input_file)
        str=os.path.join(name,name+"_seg.nii.gz")
        damar = slicer.util.loadSegmentation(os.path.join(model_dir,str)); damar.SetName('damar')
        damarLabelMap=slicer.util.loadLabelVolume(os.path.join(model_dir,str)); damarLabelMap.SetName('damarLabelMap')
        
        output_file=os.path.join(model_dir,str)
        output_folder=os.path.join(model_dir,name)
        shutil.rmtree(output_folder)
        damarCenterCoordinates=self.findCenterTumor(damar)
        
        # Add Target Markup            
        #TargetNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode","Target")        
        #n = TargetNode.AddControlPoint(damarCenterCoordinates[0], damarCenterCoordinates[1], damarCenterCoordinates[2])
        #TargetNode.SetNthControlPointLabel(n, "Target")
        

        '''utils = {
            "ijkToRAS":slicer.util.vtkMatrixFromArray(mri_image['image'].meta['original_affine']),
            "np_data":np.transpose(test_outputs, (2, 0, 1)).astype('float32')
        }

        volumeNode = slicer.util.addVolumeFromArray(utils['np_data'], 
                                            ijkToRAS=utils['ijkToRAS'], nodeClassName="vtkMRMLLabelMapVolumeNode")
        volumeNode.SetName('vessel_labelmap')

        colorNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLColorTableNode')
        colorNode.SetTypeToUser()
        colorNode.SetNumberOfColors(2) # Etiket 0 (arkaplan) ve 1 (etiket) için iki renk
        colorNode.SetColor(0, 0, 0, 0, 0) # Arkaplanı şeffaf yap
        colorNode.SetColor(1, 1, 0, 0, 1) # Etiket 1 için kırmızı renk
        colorNode.SetName('MyRedColorTable')

        # Etiket haritasına renk haritasını atayın
        volumeNode.GetDisplayNode().SetAndObserveColorNodeID(colorNode.GetID())'''

       
        



 
        #ROI oluştur

    def ProcessRiskHesapla(self):
        logging.info('Risk hesaplama yazılımı başladı...')

    def cutUpperShellAtlas(self):
        logging.info("Atlas kesiliyor...")
        #AtlasBrainM = slicer.util.getFirstNodeByClassByName("vtkMRMLScalarVolumeNode", "AtlasBrainM")    
        AtlasBrainSeg = slicer.util.getFirstNodeByClassByName("vtkMRMLSegmentationNode", "AtlasBrainSeg")
        
        if AtlasBrainSeg:    
            
            # MRABrainSeg ==> MRABrainLM
            segs = vtk.vtkStringArray(); 
            AtlasBrainSeg.GetDisplayNode().GetVisibleSegmentIDs(segs)
            AtlasBrainLM = slicer.vtkMRMLLabelMapVolumeNode()
            AtlasBrainLM.SetName("AtlasBrainLM")
            slicer.mrmlScene.AddNode(AtlasBrainLM)
            slicer.vtkSlicerSegmentationsModuleLogic.ExportAllSegmentsToLabelmapNode(AtlasBrainSeg,AtlasBrainLM) 


            # MRABrainLMShell = BinaryThinningImageFilter(MRABrainLM)            
            filter = sitk.BinaryContourImageFilter()
            filter.SetBackgroundValue(0.0)
            filter.SetDebug(False)
            filter.SetForegroundValue(1.0)
            filter.SetFullyConnected(True)
            filter.SetNumberOfThreads(20)
            filter.SetNumberOfWorkUnits(0)   
            AtlasBrainLMShell = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "AtlasBrainSegTopShellM")
            sitkUtils.PushVolumeToSlicer(filter.Execute(sitkUtils.PullVolumeFromSlicer(AtlasBrainLM)),AtlasBrainLMShell)
            
            # MRABrainLMShell ==> MRABrainShellSeg
            AtlasBrainShellSeg = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode","AtlasBrainSegShellSeg")
            slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(AtlasBrainLMShell, AtlasBrainShellSeg)
            AtlasBrainShellSeg.CreateClosedSurfaceRepresentation()   

            # MRABrainShellSeg ==> MRABrainTopShellM
            shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
            exportFolderItemId = shNode.CreateFolderItem(shNode.GetSceneItemID(), "AtlasBrainSegTopShellM")
            slicer.modules.segmentations.logic().ExportAllSegmentsToModels(AtlasBrainShellSeg, exportFolderItemId)            
            
            # Get TargetP from MRABrainTopShellM
            AtlasBrainSegTopShellM = slicer.util.getFirstNodeByClassByName("vtkMRMLModelNode", "AtlasBrainSegTopShellM")
            AtlasBrainM = slicer.util.getFirstNodeByClassByName("vtkMRMLModelNode", "AtlasBrainM")
            pointCoordinates = slicer.util.arrayFromModelPoints(AtlasBrainM)     
            TargetP = np.average(pointCoordinates, axis=0)
                            
            # Shift Camera to TargetP
            for sliceNode in slicer.util.getNodesByClass('vtkMRMLSliceNode'):
                sliceNode.JumpSliceByCentering(*TargetP)
            for camera in slicer.util.getNodesByClass('vtkMRMLCameraNode'):
                camera.SetFocalPoint(TargetP)    

            # Add Target Markup            
            TargetNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode","Target")        
            n = TargetNode.AddControlPoint(TargetP[0], TargetP[1], TargetP[2])
            TargetNode.SetNthControlPointLabel(n, "Target")
            
            # BeyinDisModel'in ust kismini kes (BeyinDisUstModel)
            redSliceNode = slicer.util.getFirstNodeByClassByName("vtkMRMLSliceNode", "Red")                                
            DynamicModeler = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLDynamicModelerNode",'DynamicModeler')
            DynamicModeler.SetToolName("Plane cut")                
            DynamicModeler.SetNodeReferenceID("PlaneCut.InputModel", AtlasBrainM.GetID())                
            DynamicModeler.SetNodeReferenceID("PlaneCut.InputPlane", redSliceNode.GetID())
            DynamicModeler.SetNodeReferenceID("PlaneCut.OutputPositiveModel", AtlasBrainM.GetID())
            slicer.modules.dynamicmodeler.logic().RunDynamicModelerTool(DynamicModeler)   
            slicer.mrmlScene.RemoveNode(DynamicModeler)
            slicer.mrmlScene.RemoveNode(AtlasBrainShellSeg)
            slicer.mrmlScene.RemoveNode(AtlasBrainLM)
            slicer.mrmlScene.RemoveNode(AtlasBrainLMShell)   
            slicer.mrmlScene.RemoveNode(TargetNode)          
            
                        
                    
        else:
            slicer.util.messageBox("You should load  AtlasBrain  Segmentation first")

    def MRABrainTopShellButton(self):
        MRA = slicer.util.getFirstNodeByClassByName("vtkMRMLScalarVolumeNode", "MRA")    
        MRABrainSeg = slicer.util.getFirstNodeByClassByName("vtkMRMLSegmentationNode", "MRABrainSeg")
        
        if MRA and MRABrainSeg:    
            
            # MRABrainSeg ==> MRABrainLM
            segs = vtk.vtkStringArray(); 
            MRABrainSeg.GetDisplayNode().GetVisibleSegmentIDs(segs)
            MRABrainLM = slicer.vtkMRMLLabelMapVolumeNode()
            MRABrainLM.SetName("MRABrainLM")
            slicer.mrmlScene.AddNode(MRABrainLM)
            slicer.vtkSlicerSegmentationsModuleLogic.ExportSegmentsToLabelmapNode(MRABrainSeg, segs, MRABrainLM, MRA)

            # MRABrainLMShell = BinaryThinningImageFilter(MRABrainLM)            
            filter = sitk.BinaryContourImageFilter()
            filter.SetBackgroundValue(0.0)
            filter.SetDebug(False)
            filter.SetForegroundValue(1.0)
            filter.SetFullyConnected(True)
            filter.SetNumberOfThreads(20)
            filter.SetNumberOfWorkUnits(0)   
            MRABrainLMShell = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "MRABrainTopShellM")
            sitkUtils.PushVolumeToSlicer(filter.Execute(sitkUtils.PullVolumeFromSlicer(MRABrainLM)),MRABrainLMShell)
            
            # MRABrainLMShell ==> MRABrainShellSeg
            MRABrainShellSeg = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode","MRABrainShellSeg")
            slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(MRABrainLMShell, MRABrainShellSeg)
            MRABrainShellSeg.CreateClosedSurfaceRepresentation()   

            # MRABrainShellSeg ==> MRABrainTopShellM
            shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
            exportFolderItemId = shNode.CreateFolderItem(shNode.GetSceneItemID(), "MRABrainTopShellM")
            slicer.modules.segmentations.logic().ExportAllSegmentsToModels(MRABrainShellSeg, exportFolderItemId)            
            
            # Get TargetP from MRABrainTopShellM
            MRABrainTopShellM = slicer.util.getFirstNodeByClassByName("vtkMRMLModelNode", "MRABrainTopShellM")
            pointCoordinates = slicer.util.arrayFromModelPoints(MRABrainTopShellM)     
            TargetP = np.average(pointCoordinates, axis=0)
                            
            # Shift Camera to TargetP
            for sliceNode in slicer.util.getNodesByClass('vtkMRMLSliceNode'):
                sliceNode.JumpSliceByCentering(*TargetP)
            for camera in slicer.util.getNodesByClass('vtkMRMLCameraNode'):
                camera.SetFocalPoint(TargetP)    

            # Add Target Markup            
            TargetNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode","Target")        
            n = TargetNode.AddControlPoint(TargetP[0], TargetP[1], TargetP[2])
            TargetNode.SetNthControlPointLabel(n, "Target")
            
            # BeyinDisModel'in ust kismini kes (BeyinDisUstModel)
            redSliceNode = slicer.util.getFirstNodeByClassByName("vtkMRMLSliceNode", "Red")                                
            DynamicModeler = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLDynamicModelerNode",'DynamicModeler')
            DynamicModeler.SetToolName("Plane cut")                
            DynamicModeler.SetNodeReferenceID("PlaneCut.InputModel", MRABrainTopShellM.GetID())                
            DynamicModeler.SetNodeReferenceID("PlaneCut.InputPlane", redSliceNode.GetID())
            DynamicModeler.SetNodeReferenceID("PlaneCut.OutputPositiveModel", MRABrainTopShellM.GetID())
            slicer.modules.dynamicmodeler.logic().RunDynamicModelerTool(DynamicModeler)   
            slicer.mrmlScene.RemoveNode(DynamicModeler)
            slicer.mrmlScene.RemoveNode(MRABrainShellSeg)
            slicer.mrmlScene.RemoveNode(MRABrainLM)
            slicer.mrmlScene.RemoveNode(MRABrainLMShell)            

        else:
            slicer.util.messageBox("You should complete MRA Brain Segmentation first")


    def create_labelmap_from_slicer(self,brain_model, vessel_model, target_coords, volume_shape=(500, 500, 500)):
        """
        Slicer'dan alınan PolyData nesnelerini kullanarak labelmapVolume oluşturur.

        :param brain_model: PyVista PolyData nesnesi olarak beyin modeli (Slicer'dan alınmış).
        :param vessel_model: PyVista PolyData nesnesi olarak damar modeli (Slicer'dan alınmış).
        :param target_coords: Hedef koordinatlarının bir listesi (örneğin, [(x1, y1, z1), (x2, y2, z2)]).
        :param volume_shape: 3D grid boyutu (varsayılan: (240, 240, 240)).
        :return: labelmapVolume (NumPy array).
        """
        
        import numpy as np
        import pyvista as pv
        # Boş bir 3D array (labelmap) oluşturuyoruz
        labelmap_volume = np.zeros(volume_shape, dtype=np.int32)

        # Beyin modelinin noktalarını alıp her nokta için etiket atama
        brain_points = brain_model.GetPoints()
        brain_points = np.array([brain_points.GetPoint(i) for i in range(brain_points.GetNumberOfPoints())])
        offset = np.abs(brain_points.min(axis=0))
        points_shifted = brain_points + offset  # Tüm noktaları pozitif yap

        # Noktaları en yakın tam sayıya yuvarlama
        grid_points = np.round(points_shifted).astype(int)
        for point in grid_points:
            # Beyin modeli 1 değeriyle etiketleniyor
            x, y, z = np.round(point).astype(int)  # İndeks olarak kullanabilmek için yuvarlıyoruz
            if 0 <= x < volume_shape[0] and 0 <= y < volume_shape[1] and 0 <= z < volume_shape[2]:
                labelmap_volume[x, y, z] = 1

        # Damar modelinin noktalarını alıp her nokta için etiket atama
        vessel_points = vessel_model.GetPoints()
        vessel_points = np.array([vessel_points.GetPoint(i) for i in range(vessel_points.GetNumberOfPoints())])

        offset = np.abs(vessel_points.min(axis=0))
        points_shifted = vessel_points + offset  # Tüm noktaları pozitif yap

        # Noktaları en yakın tam sayıya yuvarlama
        grid_points = np.round(points_shifted).astype(int)
        for point in points_shifted:
            # Damar modeli 3 değeriyle etiketleniyor
            x, y, z = np.round(point).astype(int)
            if 0 <= x < volume_shape[0] and 0 <= y < volume_shape[1] and 0 <= z < volume_shape[2]:
                labelmap_volume[x, y, z] = 3

        # Hedef koordinatlarını etiketleme



        # Noktaları en yakın tam sayıya yuvarlama
        grid_points = np.round(points_shifted).astype(int)
        x, y, z = np.round(target_coords).astype(int)
        if 0 <= x < volume_shape[0] and 0 <= y < volume_shape[1] and 0 <= z < volume_shape[2]:
            labelmap_volume[x, y, z] = 2

        count_2s = np.count_nonzero(labelmap_volume == 3)
        print(count_2s)

        return labelmap_volume




    def combine(self,shell_model,vessel_model,target_coord):
        import numpy as np
        import pyvista as pv
        import SimpleITK as sitk
 
        # Pyvista'da nokta bulutu oluştur
        target_model = pv.PolyData(target_coord)
        target_model["scalars"] = np.ones(1)  # Her nokta için etiket

        # Kürelerle temsil etmek isterseniz:
        sphere_radius = 2.0  # Kürelerin yarıçapı
        spheres = []
       
        sphere = pv.Sphere(center=target_coord, radius=sphere_radius)
        spheres.append(sphere)

        # Sferik model birleştirme
        target_sphere_model = pv.MultiBlock(spheres).combine()



        # 1. Hacim boyutlarını tanımlayın
        volume_shape = (240, 240, 240)  # 3D hacim boyutu
        labelmap_array = np.zeros(volume_shape, dtype=np.uint8)  # Boş labelmap

        # 2. VTK ve VTP dosyalarını okuyun
        #shell_model = pv.read("shell.vtk")  # Üst kabuk modeli
        #vessel_model = pv.read("vessel.vtp")  # Damar modeli
        #target_model = pv.read("target.vtk")  # Hedef modeli
        # Grid boyutlarını ve çözünürlüğü tanımlayın
        dimensions = (240, 240, 240)
        spacing = (1.0, 1.0, 1.0)  # Voxel boyutları

        # Grid noktalarını numpy ile oluştur
        x = np.linspace(0, (dimensions[0]-1)*spacing[0], dimensions[0])
        y = np.linspace(0, (dimensions[1]-1)*spacing[1], dimensions[1])
        z = np.linspace(0, (dimensions[2]-1)*spacing[2], dimensions[2])

        # Noktaları PyVista'nın StructuredGrid'ine dönüştür
        grid = pv.StructuredGrid(*np.meshgrid(x, y, z, indexing="ij"))
        # 3. Voxel ızgarası oluşturun
        #grid = pv.UniformGrid()
        grid.dimensions = volume_shape
        grid.origin = (0, 0, 0)  # Orijin
        grid.spacing = (1, 1, 1)  # Voxel boyutları



        # 4. Modelleri Voxel Hacmine Dönüştür ve Etiket Ata
        # Üst kabuk için
        shell_polydata = shell_model.GetPolyData()
        shell_pv = pv.wrap(shell_polydata)
        # PyVista uniform grid oluştur
        # Shell modeli grid üzerinde örnekle
        shell_voxels = shell_pv.sample(grid)
        labelmap_array_flat = labelmap_array.flatten()

        # Create an empty full mask with the shape of labelmap_array
        full_mask = np.zeros(labelmap_array.shape, dtype=bool)
        print(type(shell_voxels))
        # Assuming vtkValidPointMask is a list of indices, you could map them like this:
        full_mask[shell_voxels["vtkValidPointMask"]] = True

        # Now apply the mask to the full volume
        labelmap_array[full_mask] = 1

        count_label_2 = np.sum(labelmap_array == 1)
        print(count_label_2)
        print(full_mask.shape)

        #labelmap_array_flat[shell_voxels["vtkValidPointMask"] > 0] = 1
        

        # Target noktalarını labelmap'e dönüştür
        target_polydata = target_model
        shell_pv = pv.wrap(target_polydata)
        target_voxels = target_sphere_model.sample(grid)

        # Create an empty full mask with the shape of labelmap_array
        full_mask = np.zeros(labelmap_array.shape, dtype=bool)
        full_mask[target_voxels["vtkValidPointMask"]] = True

        # Now apply the mask to the full volume
        labelmap_array[full_mask] = 2
        # 5. NumPy dizisini SimpleITK formatına dönüştür
        #labelmap_array[target_voxels["vtkValidPointMask"] > 0] = 2  # Target etiketi



        # Damar için
        vessel_polydata = vessel_model.GetPolyData()
        vessel_pv = pv.wrap(vessel_polydata)
        vessel_voxels = vessel_pv.sample(grid)


        # Create an empty full mask with the shape of labelmap_array
        full_mask = np.zeros(labelmap_array.shape, dtype=bool)
        full_mask[vessel_voxels["vtkValidPointMask"]] = True

        # Now apply the mask to the full volume
        labelmap_array[full_mask] = 3
        #labelmap_array[vessel_voxels["vtkValidPointMask"] > 0] = 3


        labelmap_image = sitk.GetImageFromArray(labelmap_array)
        labelmap_image.SetSpacing((1.0, 1.0, 1.0))  # Voxel boyutu
        labelmap_image.SetOrigin((0.0, 0.0, 0.0))  # Orijin
        path=r"C:\\Users\\mustafa\\Desktop\\combined_labelmap.nrrd"
        # 6. Labelmap'i Kaydet
        sitk.WriteImage(labelmap_image, path)
        print("Labelmap başarıyla oluşturuldu: combined_labelmap.nrrd")

    def createCombinedVolumeWithTarget(self,brainModelNode, veinModelNode, targetNode):
        """
        MRABrainTop ve VeinM modelleri ile birleştirilmiş ve Target noktasını içeren label map volume oluşturur.
        Ayrıca, MRABrainTop ve VeinM modellerine etiket değerleri atanır.
        """

        # 1. Model sınırlarını alın
        brainBounds = [0] * 6
        veinBounds = [0] * 6
        brainModelNode.GetBounds(brainBounds)
        veinModelNode.GetBounds(veinBounds)
        
        # 2. En büyük boyutu belirleyin (x, y, z eksenlerinde)
        combinedMin = np.minimum(brainBounds[::2], veinBounds[::2])  # En küçük koordinatlar
        combinedMax = np.maximum(brainBounds[1::2], veinBounds[1::2])  # En büyük koordinatlar
        
        # 3. Hacim boyutlarını hesaplayın
        combinedDimensions = np.ceil((combinedMax - combinedMin)).astype(int)
        
        # 4. Hacim için voxel boyutlarını belirleyin (spacing)
        spacing = [1, 1, 1]  # Varsayılan voxel boyutu (1 mm)
        
        # 5. Yeni label map node'u oluşturun
        combinedVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "CombinedLabelMap")
        combinedVolumeNode.SetSpacing(spacing)
        combinedVolumeNode.SetOrigin(combinedMax)
        
        # 6. Segmentasyon node'u oluşturun
        segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "CombinedSegmentation")
        segmentationLogic = slicer.modules.segmentations.logic()
        # Modelleri segmentasyona ekle
        segmentationLogic.ImportModelToSegmentationNode(brainModelNode, segmentationNode)
        segmentationLogic.ImportModelToSegmentationNode(veinModelNode, segmentationNode)
        # MRABrainTop için etiketi 1 olarak ayarla
        brainSegmentId = segmentationNode.GetSegmentation().GetNthSegmentID(0)  # İlk segment
        segmentationNode.GetSegmentation().GetSegment(brainSegmentId).SetTag("LabelValue", "1")
        
        # VeinM için etiketi 3 olarak ayarla
        veinSegmentId = segmentationNode.GetSegmentation().GetNthSegmentID(1)  # İkinci segment
        segmentationNode.GetSegmentation().GetSegment(veinSegmentId).SetTag("LabelValue", "3")
   
        # 10. Target koordinatını etiketi 2 olarak ekleyin
        targetPosition = [0, 0, 0]
        targetNode.GetNthControlPointPosition(0, targetPosition)

        # Target koordinatı için bir model oluştur
        targetPolyData = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        points.InsertNextPoint(targetPosition)
        targetPolyData.SetPoints(points)
        
        vertex = vtk.vtkVertex()
        vertex.GetPointIds().SetId(0, 0)
        cells = vtk.vtkCellArray()
        cells.InsertNextCell(vertex)
        targetPolyData.SetVerts(cells)
        
        # Target model node'u oluştur
        targetModelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "TargetModel")
        targetModelNode.SetAndObservePolyData(targetPolyData)
        targetModelNode.CreateDefaultDisplayNodes()
        targetModelNode.GetDisplayNode().SetColor(1, 0, 0)  # Kırmızı renk
        
        # Target modelini segmentasyona ekle
        segmentationLogic.ImportModelToSegmentationNode(targetModelNode, segmentationNode)
        targetSegmentId = segmentationNode.GetSegmentation().GetNthSegmentID(2)  # Üçüncü segment
        segmentationNode.GetSegmentation().GetSegment(targetSegmentId).SetTag("LabelValue", "2")
        # Tüm segmentleri birleştirerek label map oluştur
        #combinedLabelMap = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "CombinedLabelMap")
        segmentationLogic.ExportAllSegmentsToLabelmapNode(segmentationNode,combinedVolumeNode)
        
        
        # 14. Güncellenmiş array'i label map'e yazın
        #slicer.util.updateVolumeFromArray(combinedVolumeNode, combinedArray)
        
        # 15. Segmentasyonu ve modelleri doğru şekilde görselleştirin
        slicer.util.setSliceViewerLayers(background=combinedVolumeNode)

        return combinedVolumeNode

    def bresenham_line_3d_float(self,start, end):
        """
        Generate points on a 3D line using Bresenham's algorithm for float coordinates.

        Parameters:
        - start: Tuple (x1, y1, z1) representing the start point (float allowed).
        - end: Tuple (x2, y2, z2) representing the end point (float allowed).

        Returns:
        - A list of points (x, y, z) along the line.
        """
        x1, y1, z1 = start
        x2, y2, z2 = end

        # Calculate differences
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        dz = abs(z2 - z1)

        # Determine the step direction
        sx = 1 if x2 > x1 else -1
        sy = 1 if y2 > y1 else -1
        sz = 1 if z2 > z1 else -1

        # Determine the dominant axis
        if dx >= dy and dx >= dz:  # X-major
            steps = int(round(dx))
        elif dy >= dx and dy >= dz:  # Y-major
            steps = int(round(dy))
        else:  # Z-major
            steps = int(round(dz))

        # Compute step increments
        x_inc = (x2 - x1) / steps
        y_inc = (y2 - y1) / steps
        z_inc = (z2 - z1) / steps

        # Generate points along the line
        points = []
        x, y, z = x1, y1, z1
        for _ in range(steps + 1):
            points.append((round(x), round(y), round(z)))  # Round to nearest integer
            x += x_inc
            y += y_inc
            z += z_inc

        return points
    
    def MRAVeinRiskButton(self):  
              
        MRABrainTopShellM = slicer.util.getFirstNodeByClassByName("vtkMRMLModelNode", "MRABrainTopShellM")
        TargetN = slicer.util.getFirstNodeByClassByName("vtkMRMLMarkupsFiducialNode", "Target")
        if MRABrainTopShellM and TargetN: 
            r1=2
            r2=10                                     
            VeinM = slicer.util.loadModel('E:\\Datasets\\Damar\\00.ITKTubeTK\\Normal-0'+no+'\\AuxillaryData\\Damar.vtp');VeinM.SetName("VeinM")            
            VeinM.GetDisplayNode().SetVisibility(True)
            #L =  self.createCombinedVolumeWithTarget(MRABrainTopShellM, VeinM, TargetN)
            #L=self.combine(MRABrainTopShellM,VeinM,np.array(TargetN.GetNthControlPointPositionVector(0)))

            #L=self.create_labelmap_from_slicer(MRABrainTopShellM.GetPolyData(),VeinM.GetPolyData(),TargetN.GetNthControlPointPositionVector(0))
            #TargetP = torch.tensor(np.array(TargetN.GetNthControlPointPositionVector(0)), dtype=torch.float64)
            #TargetP = TargetP.to(device)
            # Get the image data from the label map node
            imageData = L.GetImageData()
            # VTK'dan NumPy array'e dönüştürme

            # Etiket 2 olan noktaların koordinatlarını bulun
            coordinates = np.argwhere(L == 2)
            # Retrieve the dimensions of the label map
            dimensions = imageData.GetDimensions()
            L = slicer.util.arrayFromVolume(L)

            print("Label map içinde bulunan eşsiz değerler:", np.unique(L))

            m,n,d=dimensions
            print("m n d:",m,n,d)
            Risk=np.zeros([m,n,d])
            # Find the indices of the target label value
            #coordinates = np.argwhere(L == 1)

            # Convert numpy array to a list of tuples
            coordinates_list_Exyz = [tuple(coord) for coord in coordinates]

            # Find the indices of the target label value
            target = np.argwhere(L == 0)
            print(np.size(target))
            target = np.argwhere(L == 2)
            print(np.size(target))

            # Convert numpy array to a list of tuples
            coordinates_list_target = [tuple(coord) for coord in target]
            for i,Exyz in enumerate(coordinates_list_Exyz):
                distance=np.zeros([m,n,d])
                trajectory=self.bresenham_line_3d_float(Exyz,TargetN.GetNthControlPointPositionVector(0)) #brethonm3 func? 
                distance[trajectory]=True
                sitk_dist = sitk.GetImageFromArray(distance)
                converted_image = sitk.Cast(sitk_dist, sitk.sitkUInt8)
                danielsson_filter = sitk.DanielssonDistanceMapImageFilter()
                D=danielsson_filter.Execute(converted_image) 
                VDall=D[L==3]
                if sum((VDall<r1))>0:
                    r=-1
                else:
                    VDlocal=VDall(VDall<r2)
                    r=sum(torch.exp(-1/2*(VDlocal/5)^2))
                Risk[Exyz]=r 
            Risk[Risk==-1]=max(Risk)+1       
            #self.ShowVeinRisk(Risk,None,MRABrainTopShellM)


        else:
            slicer.util.messageBox("You should obtain BrainTopShell and Target first")


    def MRAVeinRiskButtonEski(self):
            
            MRABrainTopShellM = slicer.util.getFirstNodeByClassByName("vtkMRMLModelNode", "MRABrainTopShellM")
            TargetN = slicer.util.getFirstNodeByClassByName("vtkMRMLMarkupsFiducialNode", "Target")
            device=torch.device("cuda:0")        
            torch.cuda.empty_cache()
            torch.backends.cuda.max_split_size_mb = 256
                              
            VeinM = slicer.util.loadModel('E:\\Datasets\\Damar\\00.ITKTubeTK\\Normal-0'+no+'\\AuxillaryData\\Damar.vtp');VeinM.SetName("VeinM")
            VeinM.GetDisplayNode().SetVisibility(False)
            EnterPAll = np.array(MRABrainTopShellM.GetPolyData().GetPoints().GetData())                       
            EnterPspt = np.array_split(EnterPAll, 500)        
            VeinM = self.ModelDecimation(VeinM,0.5)
            VeinPs = torch.tensor(np.array(VeinM.GetPolyData().GetPoints().GetData()), dtype=torch.float64)
            VeinPs = VeinPs.to(device)   
            TargetP = torch.tensor(np.array(TargetN.GetNthControlPointPositionVector(0)), dtype=torch.float64)
            TargetP = TargetP.to(device)                   
            EnterPspt = np.array_split(EnterPAll, 500)        

            
            D       = 30 #[30]        # silindir yaricapi
            Beta    = 0.999 #[0.999,0.98,0.96] 
            Beta =  0.8    
            ksi      =  3
            k        =  0.5 # 0.05mm 0.30mm  
            RiskAll  = []
            NSAll = []
            dMinsAll =[]
            dMeanAll = []
            for i,EnterPs in enumerate(EnterPspt):
                std      = (D-k)/ksi 
                EnterPs = torch.tensor(EnterPs, dtype=torch.float64)
                EnterPs = EnterPs.to(device)
                VE = VeinPs.unsqueeze(0) - EnterPs.unsqueeze(1)
                TE = (TargetP - EnterPs.unsqueeze(1))*1.15
                #TE = (TargetP - EnterPs.unsqueeze(1))
                L = torch.norm(TE, dim=2).squeeze(1)
                TENorm = torch.norm(TE, dim=2); TE = TE.squeeze(1)
                
                dotProduct=torch.matmul(VE,TE.unsqueeze(2)).squeeze(2) 
                pNorm = dotProduct / TENorm  

                d = torch.norm(torch.cross(VE,TE.unsqueeze(1)), dim=2) / TENorm
                dMins,index2=torch.min(d,dim=1)# her trajectorynin min damar mesafesini bul

                SilindirIci = (d<=D) & (pNorm <=TENorm)

                A = torch.exp(-d**2/(std**2))*SilindirIci

                Risk = torch.sum(A, dim=1) # risk hesabi

                DistanceRule = (dMins>Beta) 

                NS = (DistanceRule==False)

                NSAll.extend(NS.tolist())
                RiskAll.extend(Risk.tolist())


            RiskAllNp = np.array(RiskAll)
            
            RiskAllNpT = torch.from_numpy(RiskAllNp)
            
            NSAllNpT = (torch.from_numpy(np.array(NSAll)) > 0)        
            RiskAllNpTN = ((RiskAllNpT - RiskAllNpT.min()) / (RiskAllNpT.max() - RiskAllNpT.min()))*255
            VeinRisk = torch.where(NSAllNpT == True,255,RiskAllNpTN) # damarı kesenler 255 yapıldı
            MinRiskP = EnterPAll[RiskAllNp.argmin()]
            best=RiskAllNpT.min()
            print('best :',best.cpu().numpy())
            
            self.ShowVeinRisk(VeinRisk,MinRiskP,MRABrainTopShellM)
            #return RiskMap


        
       
    def CerrahRiskHesabı(self,device,CerrahPoint,VeinPs,TargetP):
            D       = 30 #[30]        # silindir yaricapi
            #Beta    = 0.96 #[0.999,0.98,0.96] 
            Ms=1     # 0.25mm 0.5mm 1mm       
            ksi      =  3
            k        =  2 # 0.75mm 2mm  
            Beta     =  k+Ms
            UzunlukLimiti = 100
            ortalamaUzaklikyaricapi=0
            RiskAll  = []; NSAll = [];
            dMinsAll =[]
            dMeanAll = []

            std      = (D-k)/ksi 
            EnterPs = CerrahPoint
            EnterPs = EnterPs.to(device)
            VE = VeinPs.unsqueeze(0) - EnterPs
            TE = (TargetP - EnterPs)
            TENorm = torch.norm(TE) #TE = TE.squeeze(1);

            dotProduct=torch.matmul(VE.squeeze(0),TE.unsqueeze(1)) 
            pNorm = dotProduct.squeeze(1) / TENorm  
            #print('dotProduct size:',dotProduct.size())
            #print('tE size:',TE.size())
            #print('tE size:',TENorm.cpu().numpy())
            d = torch.norm(torch.cross(VE.squeeze(0),TE.unsqueeze(0)), dim=1) / TENorm
            #print('d size:',d.size())
            #print('pNorm size:',VE.size())
            #print('TENorm size:',TE.size())            
            SilindirIci = (d<=D) & (pNorm.squeeze(0) <=TENorm)
            DamarNoktaSayisi = torch.sum(SilindirIci)
            #A = torch.abs((D-d))*SilindirIci  # yaricap - uzaklik
            MinDistToVein=d[d.argmin()]
            A = torch.exp(-d**2/(std**2))*SilindirIci 
            A=A/MinDistToVein
            #A = (1/(d**(0.01)+0.00000000000001))*SilindirIci
            #A = torch.abs((D-d))**(10)*SilindirIci 
            #DistanceRule = (d<Beta)               
            #eliminatedDistances = torch.where(DistanceRule == True,9999999999,d)
            #maxRisk=D**(10)*SilindirIci
            #maxRisk = torch.sum(maxRisk)
            #print('dMins :',dMins.cpu().numpy())
            Risk = torch.sum(A)
            #A = torch.exp(-d**2/(std**2))*SilindirIci
            #Risk=(Risk/maxRisk)*100
            print('Risk:',Risk.cpu().numpy())
            #MinDistToVein = d[eliminatedDistances.argmin()]
            #MinDistToVein = torch.min(d)
            
            #Risk = torch.sum(A, dim=1)
            #Risk,ind = torch.max(A, dim=1) # her entry noktası için maksimum risk değerine sahip değer alınır. 
            #print('Max Risk Size:',Risk.size())
            #NS = torch.sum((A>=Beta),dim=1) 
            #NS,indexes = torch.max((A>=Beta),dim=1)
            #RiskAll.extend(Risk.tolist())
            #NSAll.extend(NS.tolist())        
            return MinDistToVein.cpu().numpy(),Risk.cpu().numpy(),TENorm.cpu().numpy()
                
    def MRAVeinRiskButtonforAll(self):
        nolist=['03','04','06','08','09','10','11','12','17','18','20','21','22','23','25','26','27','33','34']
        nolist2 =['37','40','42','43','44','45','47','54','56','57','58','60','63','64','70','71','74','77','79','82','86','88'] 
        #nolist.extend(nolist2)
        nolistt=['09'] 
        #####excel tablosu oluştur
        workbook = xlsxwriter.Workbook('C:\\Users\\msahin\\Desktop\\MinTrajectoryDistances.xlsx')
        worksheet = workbook.add_worksheet()

        worksheet.write(1, 1,'Cerrah1MinVeinDistance')
        worksheet.write(1, 2,'Cerrah2MinVeinDistance')
        worksheet.write(1, 3,'ProposedMinVeinDistance')
        worksheet.write(1, 4,'Cerrah1MinRisk')
        worksheet.write(1, 5,'Cerrah2MinRisk')
        worksheet.write(1, 6,'ProposedRisk')
        worksheet.write(1, 7,'Cerrah1Length')
        worksheet.write(1, 8,'Cerrah2Length')
        worksheet.write(1, 9,'ProposedRiskLength')
        row = 1
        col = 0        
        for no in nolist:
            row=row+1
            worksheet.write(row, 0,'Hasta0'+no)
            
            torch.cuda.empty_cache()
            MRABrainTopShellM = slicer.util.loadModel('D:\\Datasets\\Damar\\ITKTubeTK\\Normal-0'+no+'\\AuxillaryData\\BeyinDisModel.ply');MRABrainTopShellM.SetName("MRABrainTopShellM");
            MRABrainTopShellM.GetDisplayNode().SetVisibility(False)
                              
            VeinM = slicer.util.loadModel('D:\\Datasets\\Damar\\ITKTubeTK\\Normal-0'+no+'\\AuxillaryData\\damar.vtp');VeinM.SetName("VeinM");
            VeinM.GetDisplayNode().SetVisibility(False)
            
            torch.backends.cuda.max_split_size_mb = 256
            device=torch.device("cuda:0")             

            EnterPAll = np.array(MRABrainTopShellM.GetPolyData().GetPoints().GetData())                       
            EnterPspt = np.array_split(EnterPAll, 500)        
            VeinM = self.ModelDecimation(VeinM,0.5)
            VeinPs = torch.tensor(np.array(VeinM.GetPolyData().GetPoints().GetData()), dtype=torch.float64)
            VeinPs = VeinPs.to(device)
            TargetN=slicer.util.loadMarkups('D:\\Datasets\\Damar\\ITKTubeTK\\Normal-0'+no+'\\AuxillaryData\\Target.mrk.json');TargetN.SetName("TargetN")
            TargetP = torch.tensor(np.array(TargetN.GetNthControlPointPositionVector(0)), dtype=torch.float64)
            TargetP = TargetP.to(device)
            CerrahPoints=slicer.util.loadMarkups('D:\\Datasets\\Damar\\ITKTubeTK\\Normal-0'+no+'\\AuxillaryData\\EntryCerrah.mrk.json');CerrahPoints.SetName("CerrahPoints")
            CerrahPoint1=torch.tensor(np.array(CerrahPoints.GetNthControlPointPositionVector(0)), dtype=torch.float64)
            CerrahPoint1 = CerrahPoint1.to(device)
            CerrahPoint2=torch.tensor(np.array(CerrahPoints.GetNthControlPointPositionVector(1)), dtype=torch.float64)
            CerrahPoint2 = CerrahPoint2.to(device) 
            
            ###hesaplama dizileri elde edildiği için nodeları kaldırabiliriz.
            slicer.mrmlScene.RemoveNode(MRABrainTopShellM)
            slicer.mrmlScene.RemoveNode(VeinM)
            slicer.mrmlScene.RemoveNode(TargetN)
            slicer.mrmlScene.RemoveNode(CerrahPoints)
            
            ###Öncelikle Cerrahların önerdiği yörüngenin damara olan min uzaklığını bul
            cerrah1MinVeinDistance,Risk1,uzunluk1=self.CerrahRiskHesabı(device,CerrahPoint1,VeinPs,TargetP)
            cerrah2MinVeinDistance,Risk2,uzunluk2=self.CerrahRiskHesabı(device,CerrahPoint2,VeinPs,TargetP)
            worksheet.write(row, col+1,str(cerrah1MinVeinDistance))
            worksheet.write(row, col+2,str(cerrah2MinVeinDistance))
            worksheet.write(row, col+4,str(Risk1))
            worksheet.write(row, col+5,str(Risk2))
            worksheet.write(row, col+7,str(uzunluk1))
            worksheet.write(row, col+8,str(uzunluk2))
            print('Risk1:',Risk1)
            print('Risk2:',Risk2)
            
            D       = 30 #[30]        # silindir yaricapi
            #Beta    = 0.96 #[0.999,0.98,0.96] 
            Ms=6     # 0.25mm 0.5mm 1mm       
            ksi      =  3
            k        =  6 # 0.75mm 2mm  
            Beta     =  k+Ms
            UzunlukLimiti = 200
            ortalamaUzaklikyaricapi=0
            RiskAll  = []; NSAll = [];
            dMinsAll =[]
            dMeanAll = []

            for i,EnterPs in enumerate(EnterPspt):
                std      = (D-k)/ksi 
                EnterPs = torch.tensor(EnterPs, dtype=torch.float64)
                EnterPs = EnterPs.to(device)
                VE = VeinPs.unsqueeze(0) - EnterPs.unsqueeze(1)
                #TE = (TargetP - EnterPs.unsqueeze(1))*1.15;
                TE = (TargetP - EnterPs.unsqueeze(1))
                L = torch.norm(TE, dim=2).squeeze(1)
                TENorm = torch.norm(TE, dim=2); TE = TE.squeeze(1);
                
                dotProduct=torch.matmul(VE,TE.unsqueeze(2)).squeeze(2) 
                pNorm = dotProduct / TENorm  

                d = torch.norm(torch.cross(VE,TE.unsqueeze(1)), dim=2) / TENorm
                dMins,index2=torch.min(d,dim=1)# her trajectorynin min damar mesafesini bul
                dMean=torch.mean(d,dim=1)# her trajectorynin ortalama mesafe degerini bul
                #print('dMins size :',dMins.size())
                #print('dMean size :',dMean.size())
                SilindirIci = (d<=D) & (pNorm <=TENorm)
                DamarNoktaSayisi = torch.sum(SilindirIci,dim=1)
                A = torch.exp(-d**2/(std**2))*SilindirIci
                #A = torch.abs((D-d))*SilindirIci  # yaricap - uzaklik 
                #A = (1/(d**(0.01)+0.00000000000001))*SilindirIci
                 

                #print('dMins :',dMins.cpu().numpy())
                Risk = torch.sum(A, dim=1) # risk hesabi

                #Risk,ind = torch.max(A, dim=1) # her entry noktasi icin maksimum risk degerine sahip deger alinir. 
                #print('Max Risk Size:',Risk.size())
                #NS = torch.sum((A>=Beta),dim=1) 
                #NS,indexes = torch.min((dMins<Beta),dim=1)
                #print('L size :',L.size())
                #print('dMean size :',dMean.size())
                #print('DamarNoktaSayisi size :',DamarNoktaSayisi.size())
                DistanceRule = (dMins>Beta) & (L<UzunlukLimiti) #& (dMean>ortalamaUzaklikyaricapi)
                #print('DistanceRule size :',DistanceRule.size())
                #print('DistanceRule  :',DistanceRule.cpu().numpy())
                NS = (DistanceRule==False)
                #print('NS :',NS.cpu().numpy())
                #RiskAll.extend(Risk.tolist())
                NSAll.extend(NS.tolist())
                RiskAll.extend(Risk.tolist())
                #dMeanAll.extend(dMean.tolist())
                dMinsAll.extend(dMins.tolist())
            

            
            NSAllNpT = (torch.from_numpy(np.array(NSAll)) > 0)
            boyut =len(NSAll)
            AllNotSelectedCase=torch.sum(NSAllNpT).item()# all not selected case 

            RiskAllNp = np.array(RiskAll)
            dMinsAllNp = np.array(dMinsAll)
            
            dMinsAllNpT = torch.from_numpy(dMinsAllNp) 
            while boyut == AllNotSelectedCase:
                  print('Hepsi Not selected oldu Betalar guncellenecek...')
                  Beta=Beta-0.2
                  NSAllNpT=torch.where(dMinsAllNpT > Beta,False,NSAllNpT)
                  AllNotSelectedCase=torch.sum(NSAllNpT).item()# all not selected case 

     
            RiskAllNpT = torch.from_numpy(RiskAllNp)               
            print('RiskAllNpT:',RiskAllNpT.min().item())
            

            Mindistance = torch.where(NSAllNpT == True,RiskAllNpT.max().item(),dMinsAllNpT) 
            #print('min mesafe:',Mindistance.min().item())               
            VeinRisk = torch.where(NSAllNpT == True,RiskAllNpT.max().item(),RiskAllNpT) # damarı kesenler 255 yapıldı 
            print('VeinRisk:',VeinRisk.min().item())      
            #RiskAllNpTN = ((RiskAllNpT - RiskAllNpT.min()) / (RiskAllNpT.max()+10 - RiskAllNpT.min()))*255
            #VeinRisk = torch.where(NSAllNpT == True,255,RiskAllNpTN) # damarı kesenler 255 yapıldı
            MinRiskP = EnterPAll[VeinRisk.argmin()]
            MinRiskP = torch.tensor(MinRiskP, dtype=torch.float64)
            AutomatedMinVeinDistance,Risk3,uzunluk3=self.CerrahRiskHesabı(device,MinRiskP,VeinPs,TargetP)
            worksheet.write(row, col+3,str(AutomatedMinVeinDistance))
            worksheet.write(row, col+6,str(Risk3))
            worksheet.write(row, col+9,str(uzunluk3))
            #return VeinRisk,NSAllNpT
            #self.ShowVeinRisk(VeinRisk,MinRiskP,MRABrainTopShellM)
            #strMR = os.path.dirname(__file__)+"\Img_D"+str(D)+"_"+str(Beta)+"_"+str(k)+".png"
            #self.SaveMRImg2Png(strMR)
        ##dosyayı kaydet
        workbook.close()       
            
    def ShowVeinRisk(self,VeinRisk,MinRiskP,MRABrainTopShellM):                 
        PikselRengi_vtk = vtk.vtkUnsignedCharArray()
        PikselRengi_vtk.SetNumberOfComponents(3)
        PikselRengi_vtk.SetName("Colors")
        for renk in np.array(VeinRisk.tolist()):  
            PikselRengi_vtk.InsertNextTuple3(renk,renk,renk)                
        MRABrainTopShellM.GetPolyData().GetPointData().SetScalars(PikselRengi_vtk)
        MRABrainTopShellM.GetDisplayNode().SetActiveScalarName("Colors")
        MRABrainTopShellM.GetDisplayNode().SetAndObserveColorNodeID("vtkMRMLProceduralColorNodeRedGreenBlue")
        MRABrainTopShellM.GetDisplayNode().SetScalarVisibility(True)
        MRABrainTopShellM.GetDisplayNode().SetVisibility(True)
        
        # color bar yap
        colorLegendDisplayNode = slicer.modules.colors.logic().AddDefaultColorLegendDisplayNode(MRABrainTopShellM)
        colorLegendDisplayNode.SetNumberOfLabels(10)
        colorLegendDisplayNode.SetLabelFormat('%.0f')
        colorLegendDisplayNode.SetTitleText('Risk')
        abc = colorLegendDisplayNode.GetTitleTextProperty();abc.SetFontSize(16);                
        colorLegendDisplayNode.SetVisibility(True)

        # OptimalNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode","MinRisk")        
        # OptimalNode.AddControlPoint(MinRiskP,"MinRisk")        
                
        threeDView = slicer.app.layoutManager().threeDWidget(0).threeDView()
        threeDView.resetFocalPoint()    # reset the 3D view cube size and center it
        threeDView.resetCamera()        # reset camera zoom
        threeDView.rotateToViewAxis(5)  # sagittal direction        
        slicer.app.processEvents()

    def ModelDecimation(self,Model,c):
        parameters = {"inputModel": Model,
            "outputModel": Model,
            "reductionFactor": c,
            "method": "FastQuadric", 
            "boundaryDeletion": True}
        slicer.cli.runSync(slicer.modules.decimation, None, parameters)
        return Model            
        
    def GetAtlasModelButton(self):        
        # load AtlasBrainSeg
        AtlasBrainSeg = slicer.util.getFirstNodeByClassByName("vtkMRMLSegmentationNode", "AtlasBrainSeg")
        if not AtlasBrainSeg:            
            AtlasBrainSeg = slicer.util.loadSegmentation('E:\\Datasets\\BeyinAtlas\\'+AgeGroup+'\\Parcellation.nrrd')
            AtlasBrainSeg.SetName('AtlasBrainSeg')  
            AtlasBrainSeg.GetDisplayNode().SetVisibility(False)            
        #AgeGroup = 'UNC-Pediatric1year_Brain_Atlas'
        #AgeGroup = 'UNC_Adult_Brain_Atlas'
        #AgeGroup= 'UNC_Elderly_Brain_Atlas'
        #AgeGroup = 'UNC_Pediatric_Brain_Atlas'
            
        #Manually load Cutted Atlas model
        AtlasBrainM = slicer.util.loadModel('E:\\Datasets\\BeyinAtlas\\'+AgeGroup+'\\AtlasBrainM.vtk')
        AtlasBrainM.SetName("AtlasBrainM") 
        shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
        exportFolderItemId = shNode.CreateFolderItem(shNode.GetSceneItemID(), "AtlasBrainSegFold")
        slicer.modules.segmentations.logic().ExportAllSegmentsToModels(AtlasBrainSeg, exportFolderItemId)
        segmentModels = vtk.vtkCollection()
        shNode.GetDataNodesInBranch(exportFolderItemId, segmentModels)

        """
        # Get or obtain AtlasBrainM
        AtlasBrainM = slicer.util.getFirstNodeByClassByName("vtkMRMLModelNode", "AtlasBrainM")
        if not AtlasBrainM:
            # Segments ==> Models
            shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
            exportFolderItemId = shNode.CreateFolderItem(shNode.GetSceneItemID(), "AtlasBrainSegFold")
            slicer.modules.segmentations.logic().ExportAllSegmentsToModels(AtlasBrainSeg, exportFolderItemId)
            segmentModels = vtk.vtkCollection()
            shNode.GetDataNodesInBranch(exportFolderItemId, segmentModels)

            #Combine all models ==> AtlasBrainM
            combineModels = vtk.vtkAppendPolyData()
            for idIndex in range(AtlasBrainSeg.GetSegmentation().GetNumberOfSegments()):
                segment = segmentModels.GetItemAsObject(idIndex)
                combineModels.AddInputData(segment.GetPolyData())
            combineModels.Update()        
            AtlasBrainM = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode','AtlasBrainM')
            AtlasBrainM.SetAndObservePolyData(combineModels.GetOutput())
            #AtlasBrainM.GetDisplayNode().SetScalarVisibility(True)
            #AtlasBrainM.GetDisplayNode().SetVisibility(True)
        else:
            slicer.util.messageBox("Load AtlasBrainM first")           
        """
        
    def GetRoiFromBBox(self,Bbox,Name,modelNode):                
        P = [[(Bbox[0]), (Bbox[3] + Bbox[2]) / 2, (Bbox[5] + Bbox[4]) / 2],                  
            [(Bbox[1]), (Bbox[3] + Bbox[2]) / 2, (Bbox[5] + Bbox[4]) / 2],                                
            [(Bbox[1] + Bbox[0]) / 2, (Bbox[3] ), (Bbox[5] + Bbox[4]) / 2],             
            [(Bbox[1] + Bbox[0]) / 2, (Bbox[2] ), (Bbox[5] + Bbox[4]) / 2],                   
            [(Bbox[1] + Bbox[0]) / 2, (Bbox[3] + Bbox[2]) / 2, (Bbox[5])]]                     
                        
        NodeLMFromMR = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode',Name)
        boundaryPoints = []
        for i in range(5):            
            boundaryPoints.append(P[i])
            
        locator = vtk.vtkCellLocator()
        locator.SetDataSet(modelNode.GetPolyData())
        locator.BuildLocator()
        #modelDisplay = modelNode.GetDisplayNode()
        #modelDisplay.SetVisibility(True)
        i=0
        contactPoints = []
        for point in boundaryPoints:
            closest_point = [0.0, 0.0, 0.0]
            closest_point_id = vtk.mutable(0)
            sub_id = vtk.mutable(0)
            dist2 = vtk.mutable(0.0)
            locator.FindClosestPoint(point, closest_point, closest_point_id, sub_id, dist2)
            if  i==0:                               
                closest_point[2]=P[i][2]  # sagittal (yukarı)
                closest_point[1]=P[i][1]  #  anterior (ön)                
            elif  i==1:                                
                closest_point[2]=P[i][2]  # sagittal (yukarı)
                closest_point[1]=P[i][1]  #  anterior (ön)                
            elif  i==2:                                
                closest_point[2]=P[i][2]  # sagittal (yukarı)
                closest_point[0]=P[i][0]  #  right                
            elif  i==3:                                
                closest_point[2]=P[i][2]  # sagittal (yukarı)
                closest_point[0]=P[i][0]  #  right                
            elif  i==4:                                
                closest_point[0]=P[i][0] # right
                closest_point[1]=P[i][1] # anterior (ön)                
            point=closest_point
            i=i+1            
            contactPoints.append(closest_point)
        for i in range(5):
            NodeLMFromMR.AddControlPoint(contactPoints[i],Name+str(i))
                   
        return NodeLMFromMR
    
    def FindModelEndPointButton(self):
        AtlasBrainM = slicer.util.getFirstNodeByClassByName("vtkMRMLModelNode", "AtlasBrainM")
        MRABrainM = slicer.util.getFirstNodeByClassByName("vtkMRMLModelNode", "MRABrainM")
        if AtlasBrainM and MRABrainM:    
            AtlasBrainMBbox = [0,0,0,0,0,0]; AtlasBrainM.GetRASBounds(AtlasBrainMBbox)
            MRABrainMBbox = [0,0,0,0,0,0]; MRABrainM.GetRASBounds(MRABrainMBbox)
            AtlasBrainMPoints = self.GetRoiFromBBox(AtlasBrainMBbox,'Atlas',AtlasBrainM)
            MRABrainMPoints = self.GetRoiFromBBox(MRABrainMBbox,'MRA',MRABrainM)                 
        else:
            slicer.util.messageBox("Load AtlasBrainM and MRABrainM first")
    
    def RegistrationButton(self):
        #self.FindModelEndPointButton()# CMF nin tutarlı çalışması için uygun model noktalarını güncelleyerek al 
        fixedModel = slicer.util.getFirstNodeByClassByName("vtkMRMLModelNode", "MRABrainM")
        movingModel = slicer.util.getFirstNodeByClassByName("vtkMRMLModelNode", "AtlasBrainM")
        movingLandmarks = slicer.util.getFirstNodeByClassByName("vtkMRMLMarkupsFiducialNode", "Atlas")
        fixedLandmarks = slicer.util.getFirstNodeByClassByName("vtkMRMLMarkupsFiducialNode", "MRA")
        if fixedModel and movingModel and fixedLandmarks and movingLandmarks:                          
            # Widget
            surfaceRegistrationWidget = slicer.modules.surfaceregistration.widgetRepresentation()
            
            # FiducialRegistration
            fiducialRegistrationRadioButton = surfaceRegistrationWidget.findChild("QRadioButton", "fiducialRegistration")
            fiducialRegistrationRadioButton.setChecked(True)
            
            # Inputs
            inputFixedModelSelector = surfaceRegistrationWidget.findChild("qMRMLNodeComboBox", "inputFixedModelSelector");
            inputMovingModelSelector = surfaceRegistrationWidget.findChild("qMRMLNodeComboBox", "inputMovingModelSelector");
            inputFixedLandmarksSelector = surfaceRegistrationWidget.findChild("qMRMLNodeComboBox", "inputFixedLandmarksSelector");
            inputMovingLandmarksSelector = surfaceRegistrationWidget.findChild("qMRMLNodeComboBox", "inputMovingLandmarksSelector");           
            inputFixedModelSelector.setCurrentNode(fixedModel);
            inputMovingModelSelector.setCurrentNode(movingModel);
            inputFixedLandmarksSelector.setCurrentNode(fixedLandmarks);
            inputMovingLandmarksSelector.setCurrentNode(movingLandmarks);
            
            # Output            
            outTransform = slicer.vtkMRMLLinearTransformNode()
            outTransform.SetName('T')
            slicer.mrmlScene.AddNode(outTransform)
            outputTransformSelector = surfaceRegistrationWidget.findChild("qMRMLNodeComboBox","outputTransformSelector")
            outputTransformSelector.setCurrentNode(outTransform)

            computeButton = surfaceRegistrationWidget.findChild("QPushButton", "computeButton")
            computeButton.click()
            
            ###segment modelleri dönüştür
            shNode = slicer.mrmlScene.GetSubjectHierarchyNode()  
            folderItem = shNode.GetItemByName('AtlasBrainSegFold')
            folderItemID = folderItem
            segmentModels = vtk.vtkCollection()
            shNode.GetDataNodesInBranch(folderItemID, segmentModels)              
            if AgeGroup == 'UNC_Pediatric_Brain_Atlas':
                segmentNum = 25
            else: 
                segmentNum = 27
            segment = segmentModels.GetItemAsObject(0)
            if str(type(segment)) == "<class 'MRMLCorePython.vtkMRMLFolderDisplayNode'>":
                dizi=list(range(1, segmentNum))
            else:
                dizi=list(range(0, segmentNum-1)) 	
            for idIndex in dizi:
                segment = segmentModels.GetItemAsObject(idIndex)               
                segment.SetAndObserveTransformNodeID(outTransform.GetID())
                segment.Modified()
                        
            print(outTransform.GetMatrixTransformFromParent())

        else:
            slicer.util.messageBox("Load AtlasBrainM and MRABrainM first")
                                    
    def ProcessFMRIRiskHesapla(self):       
        TargetN = slicer.util.loadMarkups('E:\\Datasets\\Damar\\00.ITKTubeTK\\Normal-0'+no+'\\AuxillaryData\\Target.mrk.json');TargetN.SetName("Target")
        #TargetN = slicer.util.getFirstNodeByClassByName("vtkMRMLMarkupsFiducialNode", "Target")
        MRABrainTopShellM = slicer.util.loadModel('E:\\Datasets\\Damar\\00.ITKTubeTK\\Normal-0'+no+'\\AuxillaryData\\BeyinDisModel.ply');MRABrainTopShellM.SetName("MRABrainTopShellM");
        #### toplam risk hesabı için eklendi
        #veinRisks,NSAllNpT=self.MRAVeinRiskButton()        
        #segment modelleri al 
        shNode = slicer.mrmlScene.GetSubjectHierarchyNode()         
        folderItem = shNode.GetItemByName('AtlasBrainSegFold')
        folderItemID = folderItem
        segmentModels = vtk.vtkCollection()
        shNode.GetDataNodesInBranch(folderItemID, segmentModels)
        if MRABrainTopShellM and TargetN:
            torch.backends.cuda.max_split_size_mb = 256
            device=torch.device("cuda:0") 
            torch.cuda.empty_cache()            

            EnterPAll = np.array(MRABrainTopShellM.GetPolyData().GetPoints().GetData())                       
            EnterPspt = np.array_split(EnterPAll, 500)        
            TargetP = torch.tensor(np.array(TargetN.GetNthControlPointPositionVector(0)), dtype=torch.float64)
            TargetP = TargetP.to(device)
            
            D        = 30        # silindir yaricapi
            RiskAll  = []
            ksi      = 3
            k        = 5
            std      = (D-k)/ksi
            ### 27 segment için risk katsayısı dizisi başlangıç değerleri 1/27
            if AgeGroup == 'UNC_Pediatric_Brain_Atlas': segmentNum = 25
            else: segmentNum = 27            
            SegmentRisksC = np.full(segmentNum, 0)
            if segmentNum == 27:  
                SegmentRisksC=[0.05, 0.05, 0.04, 0.01,0.01,0.04,0.06,0.06,0.05,0.05,0.03,0.03,0.05,0.05,0.02,0.02,0.04,0.04,0.02,0.03,0.02,0.03,0.06,0.06,0.06,0.01,0.01]
            else:
                SegmentRisksC=[0.04, 0.04, 0.02, 0.01,0.05,0.05,0.03,0.05,0.04,0,0.02,0.01,0.05,0.05,0.03,0.05,0.03,0.03,0.04,0.02,0.02,0.01,0.01,0.02,0.02]

            SegmentRisksCT = torch.tensor(SegmentRisksC,dtype=torch.float64)
            SegmentRisksCT = SegmentRisksCT.to(device)
            ######### 
            #1.Yapılacaklar  silindir içindeki beyin noktalarını bulmak için beyin noktalarından beyin noktalarına uzaklıklar bulunur.
            #2.BeyinÜstKabuk model surface registration için tercih edilebilir.
            #3. SilindirIci güncellenecek
            ### kabalaştırma işleminin bir kere yapılması gerekiyor
            #segmentModels = self.ModelDecimation(segmentModels,0.9) 
                        
            for i,EnterPs in enumerate(EnterPspt):
                ## her beyin nokta kümesi için 
                EnterPs = torch.tensor(EnterPs, dtype=torch.float64)
                EnterPs = EnterPs.to(device)
                satir_sayisi, sutun_sayisi = EnterPs.size()
                SumOfBrainRisks = torch.zeros(satir_sayisi) #  entry giriş boyutunda boş tensör oluştur
                SumOfBrainRisks = SumOfBrainRisks.to(device)
                segment = segmentModels.GetItemAsObject(0)
                if str(type(segment)) == "<class 'MRMLCorePython.vtkMRMLFolderDisplayNode'>":
                    dizi=list(range(1, segmentNum))
                else:
                    dizi=list(range(0, segmentNum-1))                 
                for idIndex in dizi:
                    segment = segmentModels.GetItemAsObject(idIndex)
                    #### 27 segment modeli iççin  kabalaştırarak kullanımı gerekiyor bir defa olması için yukarı taşınacak
                    #segment = self.ModelDecimation(segment,0.9) 
                    segmentPs = torch.tensor(np.array(segment.GetPolyData().GetPoints().GetData()), dtype=torch.float64) 
                    segmentPs = segmentPs.to(device)

                    # VE yerine segment den alınan atlas verileri  BrainToEntry  kulanıldı
                    # beyin dış kabuktan targete olan yörüngelerin kabalaştırılmış beyin  hacmindeki her bir noktaya olan uzaklığı bulunur.            
                    BrainToEntry = segmentPs.unsqueeze(0) - EnterPs.unsqueeze(1)                    
                    TE = (TargetP - EnterPs.unsqueeze(1))*1.15;
                    TENorm = torch.norm(TE, dim=2); 
                    TE = TE.squeeze(1);
                
                    dotProduct=torch.matmul(BrainToEntry,TE.unsqueeze(2)).squeeze(2) 
                    pNorm = dotProduct / TENorm  

                    d = torch.norm(torch.cross(BrainToEntry,TE.unsqueeze(1)), dim=2) / TENorm
                    SilindirIci = (d<=D) & (pNorm <=TENorm)
                    A = torch.exp(-d**2/(std**2))*SilindirIci
                    #print('A size:',A.size())
                    #print('segmentPs:',segmentPs.size())
                    #print('SilindirIci:',SilindirIci.size())
                    #print('SegmentRisksCT[idIndex]:',SegmentRisksCT[idIndex].size())
                    
                    ### silindir içinde kalan kısmı fonksiyonel bölge katsayısı ile çarp ve entry düzeyinde topla
                    BrainRisks =  torch.sum(A * SegmentRisksCT[idIndex] ,dim = 1)
                    #print('BrainRisks:',BrainRisks.size())
                    #print('BrainRisks:',BrainRisks)
                    #print('BrainRisksMax:',BrainRisks.max())
                    #### her segmente ait riskler toplanır
                    SumOfBrainRisks = SumOfBrainRisks + BrainRisks
                    #print('SumOfBrainRisks:',SumOfBrainRisks.size()) 
                    #print('SumOfBrainRisks:',SumOfBrainRisks) 
                    #print('SumOfBrainRisksMax:',SumOfBrainRisks.max())                                                           
                RiskAll.extend(SumOfBrainRisks.tolist())                
            RiskAllNp = np.array(RiskAll)        
            RiskAllNpT = torch.from_numpy(RiskAllNp)
            self.tabloOlustur(RiskAllNpT,'Fonksiyonel Riskler')
            RiskAllNpTN = ((RiskAllNpT - RiskAllNpT.min()) / (RiskAllNpT.max() - RiskAllNpT.min()))*255
            MinRiskP = EnterPAll[RiskAllNp.argmin()]
            '''
            ###toplam risk için açıldı damar risk ve fonsiyonel risk değerleri toplandı
            RiskAllNpTN=0.9*veinRisks+0.1*RiskAllNpTN
            ## tekrar normalize edilir
            RiskAllNpTN = ((RiskAllNpT - RiskAllNpT.min()) / (RiskAllNpT.max()+10 - RiskAllNpT.min()))*255
            RiskAllNpTN = torch.where(NSAllNpT == True,255,RiskAllNpTN) # damarı kesenler 255 yapıldı
            #####
            '''
            self.ShowVeinRisk(RiskAllNpTN,MinRiskP,MRABrainTopShellM)
            print('RiskAllNpT:',RiskAllNpT.size())
        else:
            slicer.util.messageBox("Can not find MRABrainTopShellM or TargetN")    
    
    def SaveMRImg2Png(self,strMR):
        if os.path.exists(strMR):
          os.remove(strMR)
          
        threeDView = slicer.app.layoutManager().threeDWidget(0).threeDView()
        threeDView.resetFocalPoint()    # reset the 3D view cube size and center it
        threeDView.resetCamera()        # reset camera zoom
        threeDView.rotateToViewAxis(5)  # sagittal direction
        renderWindow = threeDView.renderWindow()
        renderWindow.SetAlphaBitPlanes(1)
        wti = vtk.vtkWindowToImageFilter()
        wti.SetInputBufferTypeToRGBA()
        wti.SetInput(renderWindow)
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(strMR)
        writer.SetInputConnection(wti.GetOutputPort())
        writer.Write()
    ###risklerden gelen verilere göre tablo yap    
    def tabloOlustur(self,riskler,name):
        riskSet=riskler[::2500]#heronbindeki bir data
        riskSet=riskSet.cpu().numpy()
        #####excel tablosu oluştur
        workbook = xlsxwriter.Workbook('C:\\Users\\mustafa\\Desktop\\'+name+'.xlsx')
        worksheet = workbook.add_worksheet()
        worksheet.write(1, 1,'Risk Values')
        row=2
        for risk in riskSet:
            worksheet.write(row, 1,str(risk))
            row=row+1
        ##dosyayı kaydet
        workbook.close() 
