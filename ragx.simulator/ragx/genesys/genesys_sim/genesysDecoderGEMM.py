
from collections import defaultdict
import json
import sys
from energy_model.sram_energy import sram_energy
from systolic_sim.utils import *
#sys.path.insert(0, "/home/rohan/genesysSim/")

class GenesysDecoderGEMM():
    """
        Used to parse and decode the compiler output
        *_operation.txt: Get the loops and iteration
        *_string.txt: Get the number of requests and request size per ld/st  
    """
    def __init__(self, configPath, testPath, gStats, layerType) -> None:
        #
        self.gStats = gStats
        self.testPath = testPath
        self.configPath = configPath
        self.layerType = layerType
        self.freq = 200
        self.infBandwidth = 200
        self.infFrequency = 1000
        self.infLatency = 15000
        self.arrayN = 8
        self.arrayM = 8
        self.IBUFBankDepth = 1024
        self.OBUFBankDepth = 1024
        self.WBUFBankDepth = 1024
        self.BBUFBankDepth = 1024
        self.VMEMBankDepth = 1024
        self.strd = 0
        self.pad = 0

        self.DDRDims = {}
        self.layerName = self.layerName()

        self.IBUFTileSize = 0
        self.WBUFTileSize = 0
        self.BBUFTileSize = 0
        self.numComputeTiles = 1
        self.IBUFnumTiles = 1
        self.OBUFnumTiles = 1
        self.WBUFnumTiles = 1
        self.BBUFnumTiles = 1
        
        self.numSIMDComputeTiles = 1
        self.numSIMDLoadTiles = {}
        self.numSIMDLoadTiles['vmem1'] = 1
        self.numSIMDLoadTiles['vmem2'] = 1
        self.SIMDStoreNameSpace = 'vmem1'
        
        self.IBUFtileDims = {}
        self.OBUFtileDims = {}
        self.WBUFtileDims = {}
        self.BBUFtileDims = {}
        #self.ibufNumReqs = 0
        #self.obufNumReqs = 0
        #self.wbufNumReqs = 0
        #self.bbufNumReqs = 0
        self.tileDims = {}
        self.dims = {}    
        self.ibufReuse = []
        self.obufReuse = []
        self.wbufReuse = []
        self.bbufReuse = []
        self.sysComputeLoops = 1
        self.totalInputBufSize = 0
        self.decoderCycles = 500

        self.gemm4d = False
        self.matmul = False
        self.gemm = False
        self.isGemmlayer = False
    
    def parse_json_file(self) -> None:
        _jsonPath = findFile(self.testPath, '*json.json')
        with open(_jsonPath, 'r') as f:
            _data = json.load(f)
            #print (_data['program'][0]['operation_parameters'])
            #if "4d" in _data['program'][0]['operation']:
            if len(_data['program'][0]['iterable_dimensions']) == 5:
                self.gemm4d = True
            
            elif 'gemm' == _data['program'][0]['operation'] or 'gemm_relu' == _data['program'][0]['operation'] or 'gemm_tanh' == _data['program'][0]['operation']:
                self.gemm = True

            elif 'matmul' in _data['program'][0]['operation']:
                self.matmul = True

            self.isGemmlayer = self.gemm4d or self.matmul or self.gemm

            for item in _data['program'][0]['iterable_dimensions']:
                self.DDRDims[item] = _data['program'][0]['iterable_dimensions'][item]
            
            input_key = _data['program'][0]['inputs'][0]['data_path']
            in_key = 'IBUF' in input_key
            if in_key:
                for k,v in _data['program'][0]['inputs'][0]['tiling']['IBUF'].items():
                    self.IBUFtileDims[k] = v
            input_key = _data['program'][0]['inputs'][1]['data_path']
            in_key = 'WBUF' in input_key
            if in_key:
                for k,v in _data['program'][0]['inputs'][1]['tiling']['WBUF'].items():
                    self.WBUFtileDims[k] = v   
            if len(_data['program'][0]['inputs']) > 2:
                input_key = _data['program'][0]['inputs'][2]['data_path']
                in_key = 'BBUF' in input_key
                if in_key:
                    for k,v in _data['program'][0]['inputs'][2]['tiling']['BBUF'].items():
                        self.BBUFtileDims[k] = v
            out_key = _data['program'][0]['outputs'][0]['data_path']
            out_key_1 = 'OBUF' in out_key
            if out_key_1:   
                for k,v in _data['program'][0]['outputs'][0]['tiling']['OBUF'].items():
                    self.OBUFtileDims[k] = v
            else:
                out_key = _data['program'][0]['intermediate'][0]['data_path']
                out_key_1 = 'OBUF' in out_key
                if out_key_1:   
                    for k,v in _data['program'][0]['intermediate'][0]['tiling']['OBUF'].items():
                        self.OBUFtileDims[k] = v
                

            # if "4d" in  _data['program'][0]['iterable_dimensions']:
            # else:
            # print ("ibuf dim = ", self.IBUFtileDims)
            # print ("obuf dim = ", self.OBUFtileDims)
            #print ("wbuf dim = ", self.WBUFtileDims)
            # print ("bbuf dim = ", self.BBUFtileDims)
            # print ("ddr dim = ", self.DDRDims)
                    

    def layerName(self):
        _layerName = self.testPath.split('/')[-1]
        #print (f"Simulating Test: {_layerName}")
        return _layerName

    def parse_arch_cfg(self):
        #_archCfgPath = findFile(self.testPath, '*arch_cfg.json')
        _archCfgPath = findFile(self.testPath, '../*arch_cfg.json')
        with open(_archCfgPath, 'r') as f:
            _data = json.load(f)
            self.IBUFbitwidth = _data['DATA_WIDTH']
            self.OBUFbitwidth = _data['ACC_WIDTH']
            self.WBUFbitwidth = _data['WGT_WIDTH']
            self.BBUFbitwidth = _data['BIAS_WIDTH']
            self.VMEMbitwidth = _data['ACC_WIDTH'] ## todo: needs to change from compiler
            self.IBUFBankDepth = _data['IBUF_DEPTH']
            self.OBUFBankDepth = _data['OBUF_DEPTH'] * 2
            self.WBUFBankDepth = _data['WBUF_DEPTH']
            self.BBUFBankDepth = _data['BBUF_DEPTH']
            self.VMEMBankDepth = _data['VMEM_DEPTH']
            self.VMEMBanks = _data['VMEM_BANKS']
            self.IBUFinfwidth = self.infBandwidth
            self.OBUFinfwidth = self.infBandwidth
            self.WBUFinfwidth = self.infBandwidth
            self.BBUFinfwidth = self.infBandwidth
            self.arrayN = _data['ARRAY_N']
            self.arrayM = _data['ARRAY_M']
            self.IBUFNumBanks = self.arrayN
            self.OBUFNumBanks = self.arrayM
            self.WBUFNumBanks = self.arrayM * self.arrayM
            self.BBUFNumBanks = self.arrayM
            self.simdLane = _data['SIMD_WIDTH']
        
        _configPath = self.configPath
        _archCfgPath = findFile(_configPath, 'systolic_config.json')
        with open(_archCfgPath, 'r') as f:
            _data = json.load(f)
            self.infLatency = _data["infLatency"]
            self.freq = _data['frequency']
            self.infBandwidth = _data["IBUFinfBandwidth"] 
            self.infFrequency = _data["ddr_frequency Mhz"]
            self.IBUFinfBandwidth = _data["IBUFinfBandwidth"] 
            self.IBUFperAccessReadEnergy = _data["IBUFperAccessReadEnergy"]
            self.IBUFperAccessWriteEnergy = _data["IBUFperAccessWriteEnergy"]
            self.OBUFinfBandwidth = _data["OBUFinfBandwidth"]
            self.OBUFperAccessReadEnergy = _data["OBUFperAccessReadEnergy"]
            self.OBUFperAccessWriteEnergy = _data["OBUFperAccessWriteEnergy"]
            self.WBUFinfBandwidth = _data["WBUFinfBandwidth"]
            self.WBUFperAccessReadEnergy = _data["WBUFperAccessReadEnergy"]
            self.WBUFperAccessWriteEnergy = _data["WBUFperAccessWriteEnergy"]
            self.BBUFinfBandwidth = _data["BBUFinfBandwidth"]
            self.BBUFperAccessReadEnergy = _data["BBUFperAccessReadEnergy"]
            self.BBUFperAccessWriteEnergy = _data["BBUFperAccessWriteEnergy"]
            self.numTags = _data["numTags"]
            self.decoderCycles = _data["decoderCycles"]
            self.energyPerMAC = _data["energyPerMAC"]
            self.dram_cost = _data["dram_cost"]
            self.systolic_energy = _data["systolic_energy"]
            self.simd_energy_cost = _data["simd_energy_cost"]
            self.tech_node = _data["tech_node"]
    
    def parse_instruction_file(self) -> None:
        _instrPath = findFile(self.testPath, '*string_final.txt')
        with open(_instrPath, 'r') as f:
            for line in f:
                ## To find number of tiles
                if self.gemm4d == True: # gemm4dDimMapping = { 0: 'B', 1: 'C', 2: 'M', 3: 'N', 4: 'P' }
                    if "SA_LOOP_CFG 0" in line:
                        _loop = line.split(',')
                        _loopIter = int(_loop[2].strip('\n')) + 1
                        _loop = int(_loop[1].strip())
                        if _loop < 7:
                            self.tileDims[gemm4dDimMapping[_loop]] = _loopIter
                            self.numComputeTiles = self.numComputeTiles * _loopIter
                        if _loop == 0 or _loop == 1 or _loop == 2 or _loop == 3:
                            self.IBUFnumTiles = self.IBUFnumTiles * _loopIter
                        if _loop == 0 or _loop == 1 or _loop == 2 or _loop == 4:
                            self.OBUFnumTiles = self.OBUFnumTiles * _loopIter
                        if _loop == 0 or _loop == 1 or _loop == 3 or _loop == 4:
                            self.WBUFnumTiles = self.WBUFnumTiles * _loopIter
                        if _loop == 4:
                            self.BBUFnumTiles = self.BBUFnumTiles * _loopIter

                        if _loop > 6 and _loop < 14:
                            self.sysComputeLoops = self.sysComputeLoops * _loopIter 

                elif self.gemm: # gemmDimMapping = {0: 'M', 1: 'N', 2: 'P' }
                    if "SA_LOOP_CFG 0" in line:
                        _loop = line.split(',')
                        _loopIter = int(_loop[2].strip('\n')) + 1
                        _loop = int(_loop[1].strip())
                        #print("Dims ", gemmDimMapping, _loop)
                        if _loop < 7:
                            #print("Dim ", gemmDimMapping[_loop])
                            self.tileDims[gemmDimMapping[_loop]] = _loopIter
                            self.numComputeTiles = self.numComputeTiles * _loopIter
                        if _loop == 0 or _loop == 1 or _loop == 2:
                            self.IBUFnumTiles = self.IBUFnumTiles * _loopIter
                        if _loop == 0 or _loop == 2 or _loop == 3:
                            self.OBUFnumTiles = self.OBUFnumTiles * _loopIter
                        if _loop == 2 or _loop == 3:
                            self.WBUFnumTiles = self.WBUFnumTiles * _loopIter
                        if _loop == 3:
                            self.BBUFnumTiles = self.BBUFnumTiles * _loopIter

                        self.tileDims['B'] = 1
                        self.tileDims['C'] = 1
                        
                        ## To find number of systolic compute iterations
                        if _loop > 6 and _loop < 14:
                            self.sysComputeLoops = self.sysComputeLoops * _loopIter
                                
                elif self.matmul: # matmulMapping = {0: 'B', 1: 'M', 2: 'N', 3: 'P' }
                    if "SA_LOOP_CFG 0" in line:
                        _loop = line.split(',')
                        _loopIter = int(_loop[2].strip('\n')) + 1
                        _loop = int(_loop[1].strip())
                        if _loop < 7:
                            self.tileDims[matmulMapping[_loop]] = _loopIter
                            self.numComputeTiles = self.numComputeTiles * _loopIter
                        if _loop == 0 or _loop == 1 or _loop == 2:
                            self.IBUFnumTiles = self.IBUFnumTiles * _loopIter
                        if _loop == 0 or _loop == 2 or _loop == 3:
                            self.OBUFnumTiles = self.OBUFnumTiles * _loopIter
                        if _loop == 2 or _loop == 3:
                            self.WBUFnumTiles = self.WBUFnumTiles * _loopIter
                        if _loop == 3:
                            self.BBUFnumTiles = self.BBUFnumTiles * _loopIter
                        
                        self.tileDims['C'] = 1
                        
                        ## To find number of systolic compute iterations
                        if _loop > 6 and _loop < 14:
                            self.sysComputeLoops = self.sysComputeLoops * _loopIter      

                else: # dimMapping = { 0: 'OC', 1: 'N', 2: 'IC', 3: 'KH', 4: 'KW', 5: 'OH', 6: 'OW' }
                    if "SA_LOOP_CFG 0" in line: 
                        _loop = line.split(',')
                        _loopIter = int(_loop[2].strip('\n')) + 1
                        _loop = int(_loop[1].strip())
                        if _loop < 7:
                            self.tileDims[matmulMapping[_loop]] = _loopIter
                            self.numComputeTiles = self.numComputeTiles * _loopIter
                        if _loop == 0 or _loop == 1 or _loop == 2:
                            self.IBUFnumTiles = self.IBUFnumTiles * _loopIter
                        if _loop == 0 or _loop == 2 or _loop == 3:
                            self.OBUFnumTiles = self.OBUFnumTiles * _loopIter
                        if _loop == 2 or _loop == 3:
                            self.WBUFnumTiles = self.WBUFnumTiles * _loopIter
                        if _loop == 3:
                            self.BBUFnumTiles = self.BBUFnumTiles * _loopIter
                        
                        ## To find number of systolic compute iterations
                        if _loop > 6 and _loop < 14:
                            self.sysComputeLoops = self.sysComputeLoops * _loopIter    

    def parse_SIMD_instruction_file(self) -> None:
        if "fused" in self.layerType:
            _instrPath = findFile(self.testPath, '*string_final_SIMD.txt')
        else:
            _instrPath = findFile(self.testPath, '*string_final.txt')
        _oldLoopNo1 = 0
        _oldLoopNo2 = 0
        vmem1_cntr = 0 
        vmem2_cntr = 0 
        vmem1_double_used = False
        vmem2_double_used = False
        vmem1_ld_start_cntr = 0
        vmem2_ld_start_cntr = 0
        _ldvmem1 = False
        _ldvmem2 = False
        with open(_instrPath, 'r') as f:
            for line in f:
                ## find number of SIMD tiles using the store loops
                if "ST_CONFIG_BASE_LOOP_ITER 0" in line:
                    _loop = line.split(',')
                    _loopIter = int(_loop[3].strip('\n')) + 1
                    #_loop = int(_loop[2].strip())
                    _loopNspace = _loop[1].strip()
                    #self.tileDims[dimMapping[_loop]] = _loopIter
                    self.numSIMDComputeTiles = self.numSIMDComputeTiles * _loopIter
                    self.SIMDStoreNameSpace = _loopNspace
                

                if "LD_CONFIG_BASE_LOOP_ITER 0, VMEM1, 0" in line:
                    vmem1_cntr = vmem1_cntr + 1
                    vmem1_double_used = True if vmem1_cntr == 2 else False

                if "LD_CONFIG_BASE_LOOP_ITER 0, VMEM2, 0" in line:
                    vmem2_cntr = vmem2_cntr + 1
                    vmem2_double_used = True if vmem2_cntr == 2 else False

                ## Find the reuse for VMEM1 and VMEM2
                if "LD_START 0, VMEM1" in line:
                    if vmem1_double_used:
                        vmem1_ld_start_cntr = vmem1_ld_start_cntr + 1
                        if vmem1_ld_start_cntr == 2:
                            _ldvmem1 = True
                    else:
                        _ldvmem1 = True

                if "LD_START 0, VMEM2" in line:
                    if vmem2_double_used:
                        vmem2_ld_start_cntr = vmem2_ld_start_cntr + 1
                        if vmem2_ld_start_cntr == 2:
                            _ldvmem2 = True
                    else:
                        _ldvmem2 = True
                
                if "LD_CONFIG_BASE_LOOP_ITER 0, VMEM1" in line:
                    _loop = line.split(',')
                    _loopIter = int(_loop[3].strip('\n')) + 1
                    _loop = int(_loop[2].strip())

                    if _oldLoopNo1 != _loop and _ldvmem1 == False:
                        self.numSIMDLoadTiles['vmem1'] *= _loopIter
                    _oldLoopNo1 = _loop
                    
                if "LD_CONFIG_BASE_LOOP_ITER 0, VMEM2" in line:
                    _loop = line.split(',')
                    _loopIter = int(_loop[3].strip('\n')) + 1
                    _loop = int(_loop[2].strip())

                    if _oldLoopNo2 != _loop and _ldvmem2 == False:
                        self.numSIMDLoadTiles['vmem2'] *= _loopIter
                    _oldLoopNo2 = _loop
                
            #print ("SIMD Tiles and store ns = ", self.numSIMDComputeTiles, self.SIMDStoreNameSpace)          
            #print ("SIMD Load tiles = ", self.numSIMDLoadTiles)          

    def getTileReusePerBuffer(self):
        print(self.tileDims)
        _B = self.tileDims['B'] if self.gemm4d is True else 1
        _C = self.tileDims['C'] if self.gemm4d is True else 1
        _M = self.tileDims['M']
        _N = self.tileDims['N']
        _P = 1 # self.tileDims['P'], comment out for neuroweaver code
        _oldB = -1
        _oldC = -1
        _oldM = -1
        _oldN = -1
        _oldP = -1
        #print (f"HERE {_B}, {_C}, {_M}, {_N}, {_P}")
        
        for b in range(_B):
            for c in range(_C):
                for m in range(_M):
                    for n in range(_N):
                        for p in range(_P):
                            _arr1DIndex = get5Dto1DIndex(p, n, m, c, b, _N, _M, _C, _B)
                            #print (f'array idx {_arr1DIndex}')
                            if _oldB != b:   
                                self.ibufReuse[_arr1DIndex] = 0
                                self.wbufReuse[_arr1DIndex] = 0
                                self.obufReuse[_arr1DIndex] = 0
                            if _oldC != c:
                                self.ibufReuse[_arr1DIndex] = 0
                                self.wbufReuse[_arr1DIndex] = 0
                                self.obufReuse[_arr1DIndex] = 0
                            if _oldM != m:
                                self.ibufReuse[_arr1DIndex] = 0
                                self.obufReuse[_arr1DIndex] = 0
                            if _oldN != n:
                                self.ibufReuse[_arr1DIndex] = 0
                                self.wbufReuse[_arr1DIndex] = 0
                            if _oldP != p:
                                self.wbufReuse[_arr1DIndex] = 0
                                self.bbufReuse[_arr1DIndex] = 0
                                self.obufReuse[_arr1DIndex] = 0
                            #print (f"HERE new val = {oc}, {n}, {ic}, {kh}, {kw}, {oh}, {ow}")
                            #print (f"HERE old val ={_oldOC}, {_oldN}, {_oldIC}, {_oldKH}, {_oldKW}, {_oldOH}, {_oldOW}\n")
                            _oldB = b
                            _oldC = c
                            _oldM = m
                            _oldN = n
                            _oldP = p
                            
        #print ("\n ibuf reuse Here 10:", self.ibufReuse)
        #print ("\n obuf reuse Here 10:", self.obufReuse)
        #print ("\n wbuf reuse Here 10:", self.wbufReuse)
        #print ("\n bbuf reuse Here 10:", self.bbufReuse)
    

    def updateVariable(self):
        self.ibufReuse = [1 for i in range(self.numComputeTiles + 1)]  ## +1 to account for the last tile in the compute cycle, else out of bound error
        self.obufReuse = [1 for i in range(self.numComputeTiles + 1)]  ## +1 to account for the last tile in the compute cycle, else out of bound error
        self.wbufReuse = [1 for i in range(self.numComputeTiles + 1)]  ## +1 to account for the last tile in the compute cycle, else out of bound error
        self.bbufReuse = [1 for i in range(self.numComputeTiles + 1)]  ## +1 to account for the last tile in the compute cycle, else out of bound error
        

    def computeTileSize(self, _tileDims, bitwidth):
        _tsize = 1
        for k,v in _tileDims.items():
            _tsize *= v
        _tsize *= bitwidth
        return _tsize//8 # in bytes

    def getStats(self):
        self.gStats.genesys_stats['Compiler'] = defaultdict(int)
        self.gStats.genesys_stats['Compiler']['layerName'] = self.layerName
        self.gStats.genesys_stats['Compiler']['layerType'] = self.layerType
        self.gStats.genesys_stats['Compiler']['layerClass'] = "Gemm"
        self.gStats.genesys_stats['Compiler']['DDRTiling'] = self.DDRDims
        self.gStats.genesys_stats['Compiler']['IBUFTiling'] = self.IBUFtileDims
        self.gStats.genesys_stats['Compiler']['WBUFTiling'] = self.WBUFtileDims
        self.gStats.genesys_stats['Compiler']['BBUFTiling'] = self.BBUFtileDims
        self.gStats.genesys_stats['Compiler']['OBUFTiling'] = self.OBUFtileDims
        self.gStats.genesys_stats['Compiler']['NumTiles'] = self.numComputeTiles
        self.gStats.genesys_stats['Compiler']['stride'] = self.strd
        self.gStats.genesys_stats['Compiler']['pad'] = self.pad
        self.gStats.genesys_stats['Arch'] = defaultdict(int)
        self.gStats.genesys_stats['Arch']['arrayN'] = self.arrayN
        self.gStats.genesys_stats['Arch']['arrayM'] = self.arrayM
        self.gStats.genesys_stats['Arch']['memBandwidth'] = self.IBUFinfBandwidth
        self.gStats.genesys_stats['Arch']['memLatency'] = self.infLatency
        if 'fused' in self.layerType:
            self.gStats.genesys_stats['Arch']['ibufDepth'] = self.IBUFBankDepth
            self.gStats.genesys_stats['Arch']['obufDepth'] = self.OBUFBankDepth
            self.gStats.genesys_stats['Arch']['wbufDepth'] = self.WBUFBankDepth
            self.gStats.genesys_stats['Arch']['bbufDepth'] = self.BBUFBankDepth
            self.gStats.genesys_stats['Arch']['vmem1Depth'] = self.VMEMBankDepth
        elif 'systolic' in self.layerType:
            self.gStats.genesys_stats['Arch']['ibufDepth'] = self.IBUFBankDepth
            self.gStats.genesys_stats['Arch']['obufDepth'] = self.OBUFBankDepth
            self.gStats.genesys_stats['Arch']['wbufDepth'] = self.WBUFBankDepth
            self.gStats.genesys_stats['Arch']['bbufDepth'] = self.BBUFBankDepth
            self.gStats.genesys_stats['Arch']['vmem1Depth'] = 0
        
        else:
            self.gStats.genesys_stats['Arch']['ibufDepth'] = 0
            self.gStats.genesys_stats['Arch']['obufDepth'] = 0
            self.gStats.genesys_stats['Arch']['wbufDepth'] = 0
            self.gStats.genesys_stats['Arch']['bbufDepth'] = 0
            self.gStats.genesys_stats['Arch']['vmem1Depth'] = self.VMEMBankDepth
            
        self.gStats.genesys_stats['Arch']['freq'] = self.freq
    
    def init_energy_model(self):
        # pass in values of depth, datawidth, tilesize etc. to sram_energy
        ibufM = {}
        obufM = {}
        wbufM = {}
        bbufM = {}
        vmemM = {}

        ibufM['reuseList'] = self.ibufReuse
        ibufM['banks'] = self.arrayN
        ibufM['bankDepth'] = self.IBUFBankDepth
        ibufM['dataWidth'] = self.IBUFbitwidth
        ibufM['tileSize'] = self.computeTileSize(self.IBUFtileDims, self.IBUFbitwidth)

        obufM['reuseList'] = self.obufReuse
        obufM['banks'] = self.arrayM
        obufM['bankDepth'] = self.OBUFBankDepth
        obufM['dataWidth'] = self.OBUFbitwidth
        obufM['tileSize'] = self.computeTileSize(self.OBUFtileDims, self.OBUFbitwidth)

        wbufM['reuseList'] = self.wbufReuse
        wbufM['banks'] = self.arrayN * self.arrayM
        wbufM['bankDepth'] = self.WBUFBankDepth
        wbufM['dataWidth'] = self.WBUFbitwidth
        wbufM['tileSize'] = self.computeTileSize(self.WBUFtileDims, self.WBUFbitwidth)

        bbufM['reuseList'] = self.bbufReuse
        bbufM['banks'] = self.arrayM
        bbufM['bankDepth'] = self.BBUFBankDepth
        bbufM['dataWidth'] = self.BBUFbitwidth
        bbufM['tileSize'] = self.computeTileSize(self.BBUFtileDims, self.BBUFbitwidth)

        vmemM['reuseList'] = {}
        vmemM['banks'] = self.VMEMBanks
        vmemM['bankDepth'] = self.VMEMBankDepth
        vmemM['dataWidth'] = self.VMEMbitwidth
        vmemM['tileSize'] = 1

        sysNumComputeLoops = self.sysComputeLoops

        # SIMD Tiles addition
        ######################
        ######################
        simdComputeTile = self.numSIMDComputeTiles
        simdDDRLdTiles = self.numSIMDLoadTiles
        simdDDRLdTile = self.numSIMDComputeTiles
        simdStoreNameSpace = self.SIMDStoreNameSpace

        sramModel = sram_energy(self.dram_cost, sysNumComputeLoops, self.tech_node, ibufM, obufM, wbufM, bbufM, vmemM, \
                                simdComputeTile, simdDDRLdTiles, simdDDRLdTile, simdStoreNameSpace)
        
        return sramModel
        

    def cycle(self) -> int:
        self.parse_json_file()
        self.parse_instruction_file()
        if "systolic" not in self.layerType:
            self.parse_SIMD_instruction_file()
        self.parse_arch_cfg()
        self.updateVariable()
        if 'simd' not in self.layerType: 
            self.getTileReusePerBuffer()
        self.getStats()
   
        return self.decoderCycles