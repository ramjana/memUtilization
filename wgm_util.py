
import math
import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pdb

class gfx9:
    def __init__(self, numXCDs=8, chunkSize=1, numCUsPerXCD=32, L2BytesPerXCD=2097152*2, MALLBytes=268435456):
        self.numXCDs = numXCDs
        self.numCUsPerXCD = numCUsPerXCD
        self.chunkSize = chunkSize
        self.numCUs = self.numXCDs*self.numCUsPerXCD
        self.currCU = dict()
        self.L2BytesPerXCD = L2BytesPerXCD
        self.MALLBytes = MALLBytes
        for i in range(numXCDs):
            self.currCU[i] = 0
        self.CUsAllocated = 0
        self.currXCD = 0
        self.color = {0: '#b15928', 1: '#ffff99', 2: '#6a3d9a', 3: '#cab2d6', 4: '#a6cee3', 5: '#1f78b4', 6: '#b2df8a', 7: '#33a02c', 8: '#fb9a99', 9:'#e31a1c', 10: '#fdbf6f', 11: '#ff7f00'}

    def __call__(self):
        while True:
            # Chunksize workgroups will be launched per XCD before moving to the next XCD. So if we know the # of the workgroup we are launching, we can   
            self.CUsAllocated += 1
            yield (self.currXCD, self.currCU[self.currXCD], self.color[self.currXCD])
            self.currCU[self.currXCD] = (self.currCU[self.currXCD] + 1)%self.numCUsPerXCD
            if self.CUsAllocated == self.chunkSize:
                self.currXCD = (self.currXCD + 1)%self.numXCDs
                self.CUsAllocated = 0

class WorkGroup:
    def __init__(self, m, n, new_m, new_n, xcd, cu, color, width=0.1, height=0.1):
        self.m = m
        self.n = n
        self.new_m = new_m
        self.new_n = new_n
        self.xcd = xcd
        self.cu = cu
        self.width = width
        self.height = height
        self.rect = patches.Rectangle((self.width*self.n, self.height*self.m), self.width, self.height, linewidth=1, edgecolor='k', facecolor='none')
        self.new_rect = patches.Rectangle((self.width*self.new_n, self.height*self.new_m), self.width, self.height, linewidth=1, edgecolor='k', facecolor=color, alpha=0.25)




class WorkGroupMapping:

    def __init__(self, M, N, K, WGM, GPU, MT0=64, MT1=512, DU=256, workGroupsPerCU=1, width=0.1, height=0.1, elemSize=0.5, customWGM=False,debug=False):
        self.M = M
        self.N = N
        self.K = K
        self.elemSize = elemSize
        self.WGM = WGM
        self.GPU = GPU
        self.MT0 = MT0
        self.MT1 = MT1
        self.DU = DU
        self.workGroupsPerCU = workGroupsPerCU
        self.MOverMT0 = self.M//self.MT0
        self.NOverMT1 = self.N//self.MT1
        self.KOverDU = self.K//self.DU
        self.ATileBytes = self.MT0*self.DU*self.elemSize
        self.BTileBytes = self.MT1*self.DU*self.elemSize
        self.CTileBytes = self.MT0*self.MT1*self.elemSize
        self.debug = debug
        if self.WGM != 0:
            self.numWGMSets = self.NOverMT1//self.WGM
            self.numFullWG = self.NOverMT1//self.WGM
            self.remainder = self.NOverMT1%self.WGM
        cuGen = self.GPU()
        self.width = width
        self.height = height
        self.customWGM = customWGM
        self.workGroups = dict()
        self.newWorkGroups = dict()
        for y in range(self.NOverMT1):
            for x in range(self.MOverMT0):
                wg = (x, y)
                if self.WGM != 0:
                    if self.customWGM:
                        new_wg, extras = self.getCustomNewWorkGroup(wg)
                    else:
                        new_wg, extras = self.getNewWorkGroup(wg)
                else:
                    new_wg = wg
                XCD, CU, color = next(cuGen)
                self.workGroups[wg] = WorkGroup(x, y, new_wg[0], new_wg[1], XCD, CU, color, width=self.width, height=self.height)
                self.workGroups[wg].extras = extras
        for x in range(self.MOverMT0):
            for y in range(self.NOverMT1):
                wg = (x, y)
                if self.WGM != 0:
                    if self.customWGM:
                        new_wg, extras = self.getCustomNewWorkGroup(wg)
                    else:
                        new_wg, extras = self.getNewWorkGroup(wg)
                else:
                    new_wg = wg
                new_wg = (new_wg[0], new_wg[1])
                self.newWorkGroups[new_wg] = self.workGroups[wg]
        #print(f"hit_rates(l2,mall,hbm) {self.getHitRatesFast()}")
        print(f"hit-rate(l2,mall,hbm) {self.getHitRates(self.debug)}")        

    def getNewWorkGroup(self, wg):
        wgSerial = (wg[1]%self.WGM)*self.MOverMT0 + wg[0]
        wgSet = wg[1]//self.WGM
        if wgSet < self.numFullWG: 
            X = wgSerial//self.WGM
            Y = wgSerial%self.WGM
        else:
            X = wgSerial//self.remainder
            Y = wgSerial%self.remainder
        wg_new = (X, Y + wgSet*self.WGM)
        extras = {'wgSet': wgSet, 'wgSerial': wgSerial, 'X': X, 'Y': Y}
        return wg_new, extras

    def getCustomNewWorkGroup(self, wg):
        extras = dict()
        
        numWG = self.MOverMT0*self.NOverMT1
        wgPerBlockM = 2 # WG
        wgPerBlockN = 16 # WG
        numWGPerBlock = wgPerBlockM*wgPerBlockN # WG

        numBlocksM = self.MOverMT0//wgPerBlockM # Block
        numBlocksN = self.NOverMT1//wgPerBlockN # Block
        numBlocks = numBlocksM*numBlocksN # Block        

        numWGInBlocks = numWGPerBlock*numBlocks # WG
        numWGInEdges = numWG - numWGInBlocks # WG

        blockLimM = numBlocksM*wgPerBlockM # WG
        blockLimN = numBlocksN*wgPerBlockN # WG

        # Compute the serialized workgroup ID
        wgID = wg[1]*self.MOverMT0 + wg[0] # WG
        extras['wgID'] = wgID

        if wgID < numWGInBlocks:
            # wg is a blockWG
            wgIDDechunked = (wgID%self.GPU.numXCDs)*numWGPerBlock + (wgID//self.GPU.numXCDs)
            blockID = wgIDDechunked//numWGPerBlock
            blockM = blockID%numBlocksM
            blockN = blockID//numBlocksM
            wgBlockOffsetM = blockM*wgPerBlockM
            wgBlockOffsetN = blockN*wgPerBlockN
            wgBlockID = wgIDDechunked%numWGPerBlock
            wgM = wgBlockID%wgPerBlockM
            wgN = wgBlockID//wgPerBlockM
            wg_new = (wgM + wgBlockOffsetM, wgN + wgBlockOffsetN)
        else:
            # wg is an edgeWG
            wgID -= numWGInBlocks
            numWGPerBlock = 2 # In the edge, we are only allocating the extra 2 CUs in each XCD, so the effective numWGPerBlock = 2 in this region
            wgPerBlockN = 2  # Both the extra 2 CUs in each XCD are allocated in a row, so the effective wgPerBlockN = 2 in this region
            wgIDDechunked = (wgID%self.GPU.numXCDs)*numWGPerBlock + (wgID//self.GPU.numXCDs)
            blockID = wgIDDechunked//numWGPerBlock
            blockM = 0
            blockN = blockID
            wgBlockOffsetM = numBlocksM*wgPerBlockM # The edge lies at the very top of the output
            wgBlockOffsetN = blockN*wgPerBlockN
            wgBlockID = wgIDDechunked%numWGPerBlock
            wgM = 0
            wgN = wgBlockID%wgPerBlockN
            wg_new = (wgM + wgBlockOffsetM, wgN + wgBlockOffsetN)
        extras['wgIDDechunked'] = wgIDDechunked
        return wg_new, extras

    '''
    def getCustomNewWorkGroupFull(self, wg):
        extras = dict()

        numWG = self.MOverMT0*self.NOverMT1 # WG
        numFullRounds = numWG//self.GPU.numCUs # Round
        numPartialRounds = numWG%self.GPU.numCUs # Round

        wgPerBlockM = 4 # WG/Block
        wgPerBlockN = 9 # WG/Block
        wgPerBlock = wgPerBlockM*wgPerBlockN # WG/Block

        numBlocksM = self.MOverMT0//wgPerBlockM # Block
        numBlocksN = self.NOverMT1//wgPerBlockN # Block
        numBlocks = numBlocksM*numBlocksN # Block

        numWGInBlocks = wgPerBlock*numBlocks # WG
        numWGInEdges = numWG - numWGInBlocks # WG

        blockLimM = numBlocksM*wgPerBlockM # WG
        blockLimN = numBlocksN*wgPerBlockN # WG

        # Compute the serialized workgroup ID
        wgID = wg[1]*self.MOverMT0 + wg[0] # WG

        # Compute the wgRound & the serialzied wgID within the wgRound
        wgRound = wgID//self.GPU.numCUs # Round
        wgRoundID = wgID%self.GPU.numCUs # WG

        # Check if the wg is within the blockWG area or the edgeWG area
        if wgRoundID//numWGInBlocks == 0:
            # Compute the wgBlock & the serialized wgID within the block
            wgBlock = wgRoundID//wgPerBlock # Block
            wgBlockID = wgRoundID%wgPerBlock # WG

            blockN = blockSerial//nBlocksM
            blockM = blockSerial%nBlocksM
            blockWGOffset = blockSerial*wgPerBlock
            wgWithin = wgSerialWithinRound - blockWGOffset
            wgWithinN = wgWithin//wgPerBlockN
            wgWithinM = wgWithin%wgPerBlockN
            new_wgN = wgWithinN + blockN*wgPerBlockN
            new_wgM = wgWithinM + blockM*wgPerBlockM
        else:


        return wg_new, extras
        '''

    def getHitRatesFast(self):
        # Original Hit Rate Function - correct but deos not account for finite L2 & MALL capacity 
        # Gives more or less the same answers as the correct version for simple cases but much faster
        #-----------------------------------------------------------------------------------------------
        num_wg = 0
        l2Hits = 0
        mallHits = 0
        hbmHits = 0
        l2ARows = dict()
        l2BCols = dict()
        mallARows = list()
        mallBCols = list()
        for xcd in range(self.GPU.numXCDs):
            l2ARows[xcd] = list()
            l2BCols[xcd] = list()
        for wg_tup, wg in self.workGroups.items():
            num_wg += 1     
            if wg.new_m not in l2ARows[wg.xcd]:
                l2ARows[wg.xcd].append(wg.new_m)
                if wg.new_m not in mallARows:
                    mallARows.append(wg.new_m)
                    hbmHits += 1
                else:
                    mallHits += 1
            else:
                l2Hits += 1
            if wg.new_n not in l2BCols[wg.xcd]:
                l2BCols[wg.xcd].append(wg.new_n)
                if wg.new_n not in mallBCols:
                    mallBCols.append(wg.new_n)
                    hbmHits += 1
                else:
                    mallHits += 1
            else:
                l2Hits += 1
            #print('%d: (%d,%d) '%(wg.xcd, wg.new_m, wg.new_n))
            #print('l2_hits=%d mall_hits: %d,hbm_hits: %d) '%(l2Hits,mallHits,hbmHits))
        assert (l2Hits + mallHits + hbmHits)/(2*num_wg) == 1
        return l2Hits/(2*num_wg), mallHits/(2*num_wg), hbmHits/(2*num_wg)

    def getHitRates(self, debug=True, pause=False):
        debug = debug

        numRequests = 0
        l2Hits = 0
        mallHits = 0
        hbmHits = 0
        L2 = dict()
        L2Usage = dict()
        MALL = dict()
        MALLUsage = 0
        num_ways = 16
        set_size = num_ways * 128
        num_sets = self.GPU.L2BytesPerXCD//num_ways
        num_channels = 16

        for xcd in range(self.GPU.numXCDs):
            L2[xcd] = dict()
            L2Usage[xcd] = 0

        numWorkGroups = self.MOverMT0*self.NOverMT1
        numWaves = numWorkGroups/(self.workGroupsPerCU*self.GPU.numCUs)
        numFullWaves = int(math.floor(numWaves))
        if numFullWaves != numWaves:
            partialWave = True
        else:
            partialWave = False
        workGroupsPerFullWave = self.GPU.numCUs*self.workGroupsPerCU
        workGroupsPerPartialWave = numWorkGroups - workGroupsPerFullWave*numFullWaves

        if debug:
            print('MxNxK: %dx%dx%d; MT0xMT1xDU: %dx%dx%d'%(self.M, self.N, self.K, self.MT0, self.MT1, self.DU))
            print('numWorkGroups: %d; numWaves: %f; numFullWaves: %d; partialWave: %r'%(numWorkGroups, numWaves, numFullWaves, partialWave))
            print('workGroupsPerFullWave: %d; workGroupsPerPartialWave: %d'%(workGroupsPerFullWave, workGroupsPerPartialWave))

        clk = 0

        for wave in range(numFullWaves):
            for kSlice in range(self.KOverDU):
                for wgNumWithinFullWave in range(workGroupsPerFullWave):
                    linearWG = wave*workGroupsPerFullWave + wgNumWithinFullWave
                    n = linearWG//self.MOverMT0
                    m = linearWG%self.MOverMT0
                    wg = self.workGroups[(m,n)]
                    ATile_str = 'A(%d,%d)'%(wg.new_m, kSlice)
                    BTile_str = 'B(%d,%d)'%(kSlice, wg.new_n)
                    numRequests += 2
                    A_addr = int(wg.new_m*self.MT0*self.K*self.elemSize + kSlice*self.DU*self.elemSize)
                    B_addr = int(wg.new_n*self.MT1*self.K*self.elemSize + kSlice*self.DU*self.elemSize)
                    if debug:
                        print('%d - (%d,%d) -> (%d,%d) on xcd %d cu %d'%(clk, wg.m, wg.n, wg.new_m, wg.new_n, wg.xcd, wg.cu))

                    if debug:
                        print('%d: (%d,%d) - Requesting %s'%(clk,  wg.new_m, wg.new_n, ATile_str))

                    if ATile_str not in L2[wg.xcd].keys():

                        if debug:
                            print('%d: (%d,%d) - %s not found in L2[%d]'%(clk,  wg.new_m, wg.new_n, ATile_str, wg.xcd))
                            print('%d: (%d,%d) - L2[%d] %d bytes used of %d bytes'%(clk,  wg.new_m, wg.new_n, wg.xcd, L2Usage[wg.xcd], self.GPU.L2BytesPerXCD))

                        if L2Usage[wg.xcd] + self.ATileBytes >=self.GPU.L2BytesPerXCD:

                            if debug:
                                print('%d: (%d,%d) - L2[%d] eviction required'%(clk,  wg.new_m, wg.new_n, wg.xcd))

                            lru_entry = min(L2[wg.xcd], key=L2[wg.xcd].get)
                            lru_clk = L2[wg.xcd].pop(lru_entry, None)
                            L2Usage[wg.xcd] -= self.ATileBytes

                            if debug:
                                print('%d: (%d,%d) - Evicting %s from L2[%d] since it was last used on clk %d'%(clk,  wg.new_m, wg.new_n, lru_entry, wg.xcd, lru_clk))
                                print('%d: (%d,%d) - L2[%d] %d bytes used of %d bytes'%(clk,  wg.new_m, wg.new_n, wg.xcd, L2Usage[wg.xcd], self.GPU.L2BytesPerXCD))
                        
                        set_num = A_addr>>12 & 0xff
                        channel_num = A_addr>>7 & 0xf
                        L2[wg.xcd].update({ATile_str: clk})
                        L2Usage[wg.xcd] += self.ATileBytes

                        if debug:
                            print('%d: (%d,%d) - Fetched %d bytes of %s into L2[%d]'%(clk,  wg.new_m, wg.new_n, self.ATileBytes, ATile_str, wg.xcd))
                            print('%d: (%d,%d) - L2[%d] %d bytes used of %d bytes'%(clk,  wg.new_m, wg.new_n, wg.xcd, L2Usage[wg.xcd], self.GPU.L2BytesPerXCD))

                        if ATile_str not in MALL.keys():

                            if debug:
                                print('%d: (%d,%d) - %s not found in MALL'%(clk,  wg.new_m, wg.new_n, ATile_str))
                                print('%d: (%d,%d) - MALL %d bytes used of %d bytes'%(clk,  wg.new_m, wg.new_n, MALLUsage, self.GPU.MALLBytes))

                            if MALLUsage + self.ATileBytes >= self.GPU.MALLBytes:

                                if debug:
                                    print('%d: (%d,%d) - MALL eviction required'%(clk,  wg.new_m, wg.new_n))

                                lru_entry = min(MALL, key=MALL.get)
                                lru_clk = MALL.pop(lru_entry, None)
                                MALLUsage -= self.ATileBytes

                                if debug:
                                    print('%d: (%d,%d) - Evicting %s from MALL since it was last used on clk %d'%(clk,  wg.new_m, wg.new_n, lru_entry, wg.xcd, lru_clk))
                                    print('%d: (%d,%d) - MALL %d bytes used of %d bytes'%(clk,  wg.new_m, wg.new_n, wg.xcd, MALLUsage, self.GPU.MALLBytes))

                            MALL.update({ATile_str: clk})
                            MALLUsage += self.ATileBytes

                            if debug:
                                print('%d: (%d,%d) - Fetched %d bytes of %s into MALL'%(clk,  wg.new_m, wg.new_n, self.ATileBytes, ATile_str))
                                print('%d: (%d,%d) - MALL %d bytes used of %d bytes'%(clk,  wg.new_m, wg.new_n, MALLUsage, self.GPU.MALLBytes))

                            hbmHits += 1

                            if debug:
                                print('%d: (%d,%d) - Fetched %d bytes of %s from HBM'%(clk,  wg.new_m, wg.new_n, self.ATileBytes, ATile_str))

                        else:
                            MALL[ATile_str] = clk
                            mallHits += 1

                            if debug:
                                print('%d: (%d,%d) - Fetched %d bytes of %s from MALL'%(clk,  wg.new_m, wg.new_n, self.ATileBytes, ATile_str))

                    else:
                        L2[wg.xcd][ATile_str] = clk
                        l2Hits += 1

                        if debug:
                            print('%d: (%d,%d) - Fetched %d bytes of %s from L2[%d]'%(clk,  wg.new_m, wg.new_n, self.ATileBytes, ATile_str, wg.xcd))

                    if debug:
                        print('numRequests: %d; L2Hits: %d; MALLHits: %d; HBMHits: %d'%(numRequests, l2Hits, mallHits, hbmHits))

                    clk += 1

                    if BTile_str not in L2[wg.xcd].keys():
                        if L2Usage[wg.xcd] + self.ATileBytes >= self.GPU.L2BytesPerXCD:
                            lru_entry = min(L2[wg.xcd], key=L2[wg.xcd].get)
                            lru_clk = L2[wg.xcd].pop(lru_entry, None)
                            L2Usage[wg.xcd] -= self.ATileBytes
                            if debug:
                                print('Removing entry %s from L2 since it was last used on clk %d'%(lru_entry, lru_clk))
                        L2[wg.xcd].update({BTile_str: clk})
                        L2Usage[wg.xcd] += self.ATileBytes
                        if BTile_str not in MALL.keys():
                            if MALLUsage + self.ATileBytes >= self.GPU.MALLBytes:
                                lru_entry = min(MALL, key=MALL.get)
                                lru_clk = MALL.pop(lru_entry, None)
                                MALLUsage -= self.ATileBytes
                                if debug:
                                    print('Removing entry %s from MALL since it was last used on clk %d'%(lru_entry, lru_clk))
                            MALL.update({BTile_str: clk})
                            MALLUsage += self.ATileBytes
                            hbmHits += 1
                        else:
                            MALL[BTile_str] = clk
                            mallHits += 1
                    else:
                        L2[wg.xcd][BTile_str] = clk
                        l2Hits += 1
                    
                    clk += 1

                    if debug:
                        print('numRequests: %d; L2Hits: %d; MALLHits: %d; HBMHits: %d'%(numRequests, l2Hits, mallHits, hbmHits))        
                        if pause:
                            yn = input('Pause? (y/n)')
                            if yn.lower()[0] == 'y':
                                pdb.set_trace()

        if partialWave:
            for kSlice in range(self.KOverDU):
                for wgNumWithinPartialWave in range(workGroupsPerPartialWave):
                    linearWG = numFullWaves*workGroupsPerFullWave + wgNumWithinPartialWave
                    n = linearWG//self.MOverMT0
                    m = linearWG%self.MOverMT0
                    wg = self.workGroups[(m,n)]
                    ATile_str = 'A(%d,%d)'%(wg.new_m, kSlice)
                    BTile_str = 'B(%d,%d)'%(kSlice, wg.new_n)
                    numRequests += 2

                    if ATile_str not in L2[wg.xcd].keys():
                        if L2Usage[wg.xcd] + self.ATileBytes >= self.GPU.L2BytesPerXCD:
                            lru_entry = min(L2[wg.xcd], key=L2[wg.xcd].get)
                            L2[wg.xcd].pop(lru_entry, None)
                            L2Usage[wg.xcd] -= self.ATileBytes
                        L2[wg.xcd].update({ATile_str: clk})
                        L2Usage[wg.xcd] += self.ATileBytes
                        L2[wg.xcd].update({ATile_str: clk})
                        L2Usage[wg.xcd] += self.ATileBytes
                        if ATile_str not in MALL.keys():
                            if MALLUsage + self.ATileBytes >= self.GPU.MALLBytes:
                                #MALL.pop(0)
                                #MALLUsage -= self.ATileBytes
                                lru_entry = min(MALL, key=MALL.get)
                                lru_clk = MALL.pop(lru_entry, None)
                                MALLUsage -= self.ATileBytes
                            MALL.update({ATile_str: clk})
                            MALLUsage += self.ATileBytes
                            hbmHits += 1
                        else:
                            mallHits += 1
                    else:
                        l2Hits += 1
                    
                    clk += 1

                    if BTile_str not in L2[wg.xcd].keys():
                        if L2Usage[wg.xcd] + self.ATileBytes >= self.GPU.L2BytesPerXCD:
                            lru_entry = min(L2[wg.xcd], key=L2[wg.xcd].get)
                            lru_clk = L2[wg.xcd].pop(lru_entry, None)
                            L2Usage[wg.xcd] -= self.ATileBytes
                            if debug:
                                print('Removing entry %s from L2 since it was last used on clk %d'%(lru_entry, lru_clk))
                        L2[wg.xcd].update({BTile_str: clk})
                        L2Usage[wg.xcd] += self.ATileBytes
                        if BTile_str not in MALL.keys():
                            if MALLUsage + self.ATileBytes >= self.GPU.MALLBytes:
                                lru_entry = min(MALL, key=MALL.get)
                                lru_clk = MALL.pop(lru_entry, None)
                                MALLUsage -= self.ATileBytes
                                if debug:
                                    print('Removing entry %s from MALL since it was last used on clk %d'%(lru_entry, lru_clk))
                            MALL.update({BTile_str: clk})
                            MALLUsage += self.ATileBytes
                            hbmHits += 1
                        else:
                            MALL[BTile_str] = clk
                            mallHits += 1
                    else:
                        L2[wg.xcd][BTile_str] = clk
                        l2Hits += 1
                    
                    clk += 1

        assert (l2Hits + mallHits + hbmHits)/numRequests == 1
        return l2Hits/numRequests, mallHits/numRequests, hbmHits/numRequests

    def printWorkGroups(self):
        for x in range(self.MOverMT0):
            line = ''
            for y in range(self.NOverMT1):
                wg_tup = (x,y)
                wg = self.workGroups[wg_tup]
                line += ' |(%d, %d) -> (%d, %d)|'%(wg.m, wg.n, wg.new_m, wg.new_n)
            line += '\n'
            print(line)

    def printNewWorkGroups(self):
        for x in range(self.MOverMT0):
            line = ''
            for y in range(self.NOverMT1):
                wg_tup = (x,y)
                wg = self.newWorkGroups[wg_tup]
                line += ' |(%d, %d) -> (%d, %d)|'%(wg.m, wg.n, wg.new_m, wg.new_n)
            line += '\n'
            print(line)

    def plotWorkGroups(self, plot_launch_order=True, full_annotation=False):
        fig, ax = plt.subplots(figsize=(2*self.NOverMT1, 2*self.MOverMT0))
        plt.tick_params(left = False, right = False, labelleft = False, labelbottom = False, bottom = False)
        for x in range(self.MOverMT0):
            for y in range(self.NOverMT1):
                wg_tup = (x,y)
                wg = self.workGroups[wg_tup]
                ax.add_patch(wg.rect)
                rx, ry = wg.rect.get_xy()
                cx = rx + wg.rect.get_width()/2.0
                cy = ry + wg.rect.get_height()/2.0
                if full_annotation:
                    ax.annotate('%d,%d -> %d,%d'%(wg.m, wg.n,wg.new_m,wg.new_n), (cx, cy), color='k', fontsize=8, ha='center', va='center')
                else:
                    ax.annotate('%d,%d\nXCD%d CU%d'%(wg.m,wg.n, wg.xcd, wg.cu), (cx, cy), color='k', fontsize=8, ha='center', va='center')
        if plot_launch_order:
            Ctr = 0
            for y in range(self.NOverMT1):
                for x in range(self.MOverMT0):
                    wg_tup = (x,y)
                    wg = self.workGroups[wg_tup]
                    rx, ry = wg.rect.get_xy()
                    cx = rx + wg.rect.get_width()/2.0
                    cy = ry + wg.rect.get_height()/2.0
                    if Ctr == 0:
                        cx_begin = cx
                        cy_begin = cy
                        if not (x == 0 and y == 0):
                            cx_dot_len = cx - cx_dot_begin
                            cy_dot_len = cy - cy_dot_begin
                            plt.arrow(cx_dot_begin, cy_dot_begin, cx_dot_len, cy_dot_len, length_includes_head=True, linestyle='--', width=0.005, alpha=0.25, zorder=5)
                    if Ctr == self.MOverMT0 - 1 or (x == self.MOverMT0 - 1 and y == self.NOverMT1 - 1):
                        cx_len = cx - cx_begin
                        cy_len = cy - cy_begin
                        cx_dot_begin = cx
                        cy_dot_begin = cy
                        plt.arrow(cx_begin, cy_begin, cx_len, cy_len, length_includes_head=True, width=0.005, alpha=0.25, zorder=5)
                    Ctr = (Ctr + 1)%self.MOverMT0
        plt.xlim(0,0.1*self.NOverMT1)
        plt.ylim(0,0.1*self.MOverMT0)

    def plotNewWorkGroups(self, saveFig=False, figureTag=None, plot_launch_order=True, full_annotation=False):
        fig, ax = plt.subplots(figsize=(2*self.NOverMT1, 2*self.MOverMT0))
        plt.tick_params(left = False, right = False, labelleft = False, labelbottom = False, bottom = False)
        for wg_tup, wg in self.newWorkGroups.items():
                ax.add_patch(wg.new_rect)
                rx, ry = wg.new_rect.get_xy()
                cx = rx + wg.new_rect.get_width()/2.0
                cy = ry + wg.new_rect.get_height()/2.0
                if full_annotation:
                    if self.customWGM:
                        ax.annotate('%d,%d -> %d,%d\nXCD%d CU%d\nwgID %d -> %d'%(wg.m, wg.n,wg.new_m, wg.new_n, wg.xcd, wg.cu, wg.extras['wgID'], wg.extras['wgIDDechunked']), (cx, cy), color='k', fontsize=8, ha='center', va='center')
                    else:
                        ax.annotate('%d,%d -> %d,%d\nXCD%d CU%d\n%d;%d\n%d,%d'%(wg.m, wg.n, wg.new_m, wg.new_n, wg.xcd, wg.cu, wg.extras['wgSet'], wg.extras['wgSerial'], wg.extras['X'], wg.extras['Y']), (cx, cy), color='k', fontsize=8, ha='center', va='center')
                else:
                    ax.annotate('%d,%d\nXCD%d CU%d'%(wg.new_m,wg.new_n, wg.xcd, wg.cu), (cx, cy), color='k', fontsize=8, ha='center', va='center')
        if plot_launch_order:
            WGMCtr = 0
            for y in range(self.NOverMT1):
                for x in range(self.MOverMT0):
                    wg_tup = (x,y)
                    wg = self.workGroups[wg_tup]
                    rx, ry = wg.new_rect.get_xy()
                    cx = rx + wg.new_rect.get_width()/2.0
                    cy = ry + wg.new_rect.get_height()/2.0
                    if WGMCtr == 0:
                        cx_begin = cx
                        cy_begin = cy
                        if not (x == 0 and y == 0):
                            cx_dot_len = cx - cx_dot_begin
                            cy_dot_len = cy - cy_dot_begin
                            plt.arrow(cx_dot_begin, cy_dot_begin, cx_dot_len, cy_dot_len, length_includes_head=True, linestyle='--', width=0.005, alpha=0.25, zorder=5)
                    if WGMCtr == self.WGM - 1 or (x == self.MOverMT0 - 1 and y == self.NOverMT1 - 1):
                        cx_len = cx - cx_begin
                        cy_len = cy - cy_begin
                        cx_dot_begin = cx
                        cy_dot_begin = cy
                        plt.arrow(cx_begin, cy_begin, cx_len, cy_len, length_includes_head=True, width=0.005, alpha=0.25, zorder=5)
                    WGMCtr = (WGMCtr + 1)%self.WGM
        plt.xlim(0,0.1*self.NOverMT1)
        plt.ylim(0,0.1*self.MOverMT0)
        if saveFig:
            figureName = '%dx%dx%d_HHS_NN_%dx%dx%d_WGM%d'%(self.M, self.N, self.K, self.MT0, self.MT1, self.DU, self.WGM)
            if figureTag is not None:
                figureName += '_%s'%(figureTag)
            figureName += '.png'
            
            plt.savefig(figureName, transparent=True, dpi=300, format='png')


if __name__ == '__main__':

    gpu = gfx9();
    Wgm = WorkGroupMapping(M=128,N=106496,K=8192*2,MT0=64,MT1=512,DU=512,WGM=16,GPU=gpu,customWGM=False)
    Wgm = WorkGroupMapping(M=128,N=106496,K=8192*2,MT0=128,MT1=512,DU=512,WGM=32,GPU=gpu,customWGM=False)
