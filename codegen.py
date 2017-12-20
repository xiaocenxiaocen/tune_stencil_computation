# file: codegen.py
# author: 
# date: 2017.10.19
import numpy as np
from os import system
from sys import argv
from subprocess import Popen, call, PIPE

def extractCompileInfo(kernelName):
	makeProcess = Popen("make", shell = True, stdout = PIPE, stderr = PIPE)
	makeRetStdout = makeProcess.stdout.readlines()
	makeRetStderr = makeProcess.stderr.readlines()
	import re
	spillIdx = 0
	memIdx = 0
	for idx, str_ in enumerate(makeRetStderr):
		str_ = str(str_)
		if str_.find("Function properties") != -1 and str_.find(kernelName) != -1:
			spillIdx = idx + 1
			memIdx = idx + 2
			break
	spillInfo = str(makeRetStderr[spillIdx])
	memInfo = str(makeRetStderr[memIdx])
	stack_frame = int(spillInfo[2:spillInfo.find("bytes stack frame")])
	spill_stores = spillInfo[:spillInfo.find("bytes spill stores")]
	spill_stores = int(spill_stores[spill_stores.rfind(",") + 1:])
	spill_loads = spillInfo[:spillInfo.find("bytes spill loads")]
	spill_loads = int(spill_loads[spill_loads.rfind(",") + 1:])
	
	registers = memInfo[:memInfo.find("registers")]
	registers = int(registers[registers.find("Used") + 4:])
	smem = memInfo[:memInfo.find("bytes smem")]		
	smem = smem[smem.rfind(",") + 1:]
	smem = smem[:smem.find("+")]
	smem = int(smem)
	
	ret = {"stack_frame":stack_frame, "spill_stores":spill_stores, "spill_loads":spill_loads,"registers":registers,"smem":smem}
	return ret

# python 2
#def extractCompileInfo(kernelName):
#	makeProcess = Popen("make", shell = True, stdout = PIPE, stderr = PIPE)
#	makeRetStdout = makeProcess.stdout.readlines()
#	makeRetStderr = makeProcess.stderr.readlines()
#	import re
#	spillIdx = 0
#	memIdx = 0
#	for idx, str_ in enumerate(makeRetStderr):
#		if str_.find("Function properties") != -1 and str_.find(kernelName) != -1:
#			spillIdx = idx + 1
#			memIdx = idx + 2
#			break
#	spillInfo = makeRetStderr[spillIdx]
#	memInfo = makeRetStderr[memIdx]
#	stack_frame = int(spillInfo[:spillInfo.find("bytes stack frame")])
#	spill_stores = spillInfo[:spillInfo.find("bytes spill stores")]
#	spill_stores = int(spill_stores[spill_stores.rfind(",") + 1:])
#	spill_loads = spillInfo[:spillInfo.find("bytes spill loads")]
#	spill_loads = int(spill_loads[spill_loads.rfind(",") + 1:])
#	
#	registers = memInfo[:memInfo.find("registers")]
#	registers = int(registers[registers.find("Used") + 4:])
#	smem = memInfo[:memInfo.find("bytes smem")]		
#	smem = smem[smem.rfind(",") + 1:]
#	smem = smem[:smem.find("+")]
#	smem = int(smem)
#	
#	ret = {"stack_frame":stack_frame, "spill_stores":spill_stores, "spill_loads":spill_loads,"registers":registers,"smem":smem}
#	return ret

def extractMcellsPerSec(ret):
	mcellsps = str(ret[18])
	mcellsps = mcellsps[mcellsps.find("=") + 2:]
	mcellsps = mcellsps[:mcellsps.rfind("Mpoints/s") - 1]
	return float(mcellsps)

def gridSearch(tx, ty):
	TX = tx
	TY = ty

	result = open('config_tx_%02d_ty_%02d.csv'%(TX, TY), 'w')
	result.write("TX, TY, BX, BY, DX, DY, UNROLL, Mcells/s\n")
	result.flush()	

	opt_BX = 0
	opt_BY = 0
	opt_TX = 0
	opt_TY = 0
	opt_DX = 0
	opt_UNROLL = 0

	kernelName = "fdtd3d_kernel_template"

	maxrregcount = 255
	maxRegsPerBlock = 65536
	
	RADIUS = 4

	opt_mcells = 0

	print("starting grid search...")
	for X_factor in range(1, 6):
		for Y_factor in range(1, 6):
			for UNROLL in range(1, 10):
				BX = TX * X_factor
				BY = TY * Y_factor
				
				DX = TX
				DY = TX * TY / DX			

				print("test for search parameters: BX = %d, BY = %d, TX = %d, TY = %d, DX = %d, DY = %d, UNROLL = %d"%(BX, BY, TX, TY, DX, DY, UNROLL))
			
				if not (BX % DX == 0): continue
				if not (BY % DY == 0): continue
				if DX < RADIUS: continue
				if DY < RADIUS: continue

				cmd = '''sed "s:#define BLOCK_X:#define BLOCK_X %d:g;
					s:#define BLOCK_Y:#define BLOCK_Y %d:g;
					s:#define MEM_PATTERN_X:#define MEM_PATTERN_X %d:g;
					s:#define THREAD_X:#define THREAD_X %d:g;
					s:#define THREAD_Y:#define THREAD_Y %d:g;
					s:#pragma unroll UNROLL:#pragma unroll %d:g" iso_fd_comp_template.cu > iso_fd_comp.cu'''%(BX, BY, DX, TX, TY, UNROLL)

				call(cmd, shell = True)

				compileInfo = extractCompileInfo(kernelName)
				spill_stores = compileInfo["spill_stores"]
				spill_loads = compileInfo["spill_loads"]
				registers = compileInfo["registers"]
				smem = compileInfo["smem"]
				# python 3
				print("spill stores: %d bytes, spill loads: %d bytes, registers: %d, smem: %d bytes"%(spill_stores, spill_loads, registers, smem))
				# python 2
#				print "spill stores: ", spill_stores, " bytes, spill loads: ", spill_loads, " bytes, registers: ", registers, ", smem: ", smem, " bytes"
				smem = smem / 1024.0
				if smem > 48:
					continue
				if spill_loads > 1 or spill_stores > 1:
					continue
				if registers > maxrregcount:
					continue
				if registers * TX * TY > maxRegsPerBlock:
					continue

				ret = Popen("./fdtd3d_gpu 512 512 512 0.01 0.01 0.01 10 0.001", shell = True, stdout = PIPE, bufsize = 1).stdout.readlines()
				mcellsps = extractMcellsPerSec(ret)
				print("Performance: %.2f Mcells/s"%(mcellsps))

				result.write("%d,%d,%d,%d,%d,%d,%d,%.2f\n"%(TX, TY, BX, BY, DX, DY, UNROLL, mcellsps))
				result.flush()
				
				if mcellsps > opt_mcells:
					opt_BX = BX
					opt_BY = BY
					opt_TX = TX
					opt_TY = TY
					opt_DX = DX
					opt_UNROLL = UNROLL
					opt_mcells = mcellsps

	params = {'BX': opt_BX,
		  'BY': opt_BY,
		  'TX': opt_TX,
		  'TY': opt_TY,
		  'DX': opt_DX,
		  'UNROLL': opt_UNROLL,
		  'Mcells/s': opt_mcells}
	return params

if __name__ == "__main__":
	TX = 32
	TY = 8
	optParams = gridSearch(TX, TY)
	opt_BX = optParams['BX']
	opt_BY = optParams['BY']
	opt_TX = optParams['TX']
	opt_TY = optParams['TY']
	opt_DX = optParams['DX']
	opt_UNROLL = optParams['UNROLL']
	opt_mcellsps = optParams['Mcells/s']

	cmd = '''sed "s:#define BLOCK_X:#define BLOCK_X %d:g;
		s:#define BLOCK_Y:#define BLOCK_Y %d:g;
		s:#define MEM_PATTERN_X:#define MEM_PATTERN_X %d:g;
		s:#define THREAD_X:#define THREAD_X %d:g;
		s:#define THREAD_Y:#define THREAD_Y %d:g;
		s:#pragma unroll UNROLL:#pragma unroll %d:g" iso_fd_comp_template.cu > iso_fd_comp.cu'''%(opt_BX, opt_BY, opt_DX, opt_TX, opt_TY, opt_UNROLL)

	call(cmd, shell = True)
	Popen("make", shell = True, stdout = PIPE, stderr = PIPE).wait()
