# Make backwards compatible with python 2, ignored in python 3
from __future__ import print_function

import sys
import os
import subprocess
import struct
import os.path
import string
import time
import re
import math
import operator
import numpy as np
import h5py
# import hdf5 interface
# import tables as pytb #import pytables
#import pandas as pd
import copy
from collections import deque
import itertools
import scipy.interpolate as scipyinterp
import scipy.spatial as spatial
import multiprocessing as mp
#import mpi4py as mpi
from collections import deque
import pandas as pd
#import cython
#from cython.parallel import prange, parallel

# would be good to compile these routines with cython
# try to speed up search
# cimport numpy as np

"""

Routines for reading velociraptor output

"""

"""
    IO Routines
"""


def ReadPropertyFile(basefilename, ibinary=0, iseparatesubfiles=0, iverbose=0, desiredfields=[], isiminfo=True, iunitinfo=True, iconfiginfo=True):
    """
    VELOCIraptor/STF files in various formats
    for example ascii format contains
    a header with
        filenumber number_of_files
        numhalos_in_file nnumhalos_in_total
    followed by a header listing the information contain. An example would be
        ID(1) ID_mbp(2) hostHaloID(3) numSubStruct(4) npart(5) Mvir(6) Xc(7) Yc(8) Zc(9) Xcmbp(10) Ycmbp(11) Zcmbp(12) VXc(13) VYc(14) VZc(15) VXcmbp(16) VYcmbp(17) VZcmbp(18) Mass_tot(19) Mass_FOF(20) Mass_200mean(21) Mass_200crit(22) Mass_BN97(23) Efrac(24) Rvir(25) R_size(26) R_200mean(27) R_200crit(28) R_BN97(29) R_HalfMass(30) Rmax(31) Vmax(32) sigV(33) veldisp_xx(34) veldisp_xy(35) veldisp_xz(36) veldisp_yx(37) veldisp_yy(38) veldisp_yz(39) veldisp_zx(40) veldisp_zy(41) veldisp_zz(42) lambda_B(43) Lx(44) Ly(45) Lz(46) q(47) s(48) eig_xx(49) eig_xy(50) eig_xz(51) eig_yx(52) eig_yy(53) eig_yz(54) eig_zx(55) eig_zy(56) eig_zz(57) cNFW(58) Krot(59) Ekin(60) Epot(61) n_gas(62) M_gas(63) Xc_gas(64) Yc_gas(65) Zc_gas(66) VXc_gas(67) VYc_gas(68) VZc_gas(69) Efrac_gas(70) R_HalfMass_gas(71) veldisp_xx_gas(72) veldisp_xy_gas(73) veldisp_xz_gas(74) veldisp_yx_gas(75) veldisp_yy_gas(76) veldisp_yz_gas(77) veldisp_zx_gas(78) veldisp_zy_gas(79) veldisp_zz_gas(80) Lx_gas(81) Ly_gas(82) Lz_gas(83) q_gas(84) s_gas(85) eig_xx_gas(86) eig_xy_gas(87) eig_xz_gas(88) eig_yx_gas(89) eig_yy_gas(90) eig_yz_gas(91) eig_zx_gas(92) eig_zy_gas(93) eig_zz_gas(94) Krot_gas(95) T_gas(96) Zmet_gas(97) SFR_gas(98) n_star(99) M_star(100) Xc_star(101) Yc_star(102) Zc_star(103) VXc_star(104) VYc_star(105) VZc_star(106) Efrac_star(107) R_HalfMass_star(108) veldisp_xx_star(109) veldisp_xy_star(110) veldisp_xz_star(111) veldisp_yx_star(112) veldisp_yy_star(113) veldisp_yz_star(114) veldisp_zx_star(115) veldisp_zy_star(116) veldisp_zz_star(117) Lx_star(118) Ly_star(119) Lz_star(120) q_star(121) s_star(122) eig_xx_star(123) eig_xy_star(124) eig_xz_star(125) eig_yx_star(126) eig_yy_star(127) eig_yz_star(128) eig_zx_star(129) eig_zy_star(130) eig_zz_star(131) Krot_star(132) tage_star(133) Zmet_star(134)

    then followed by data

    Note that a file will indicate how many files the total output has been split into

    Not all fields need be read in. If only want specific fields, can pass a string of desired fields like
    ['ID', 'Mass_FOF', 'Krot']
    #todo still need checks to see if fields not present and if so, not to include them or handle the error
    """
    # this variable is the size of the char array in binary formated data that stores the field names
    CHARSIZE = 40

    start = time.process_time()
    inompi = True
    if (iverbose):
        print("reading properties file", basefilename)
    filename = basefilename+".properties"
    # load header
    if (os.path.isfile(filename) == True):
        numfiles = 0
    else:
        filename = basefilename+".properties"+".0"
        inompi = False
        if (os.path.isfile(filename) == False):
            print("Could not find VELOCIraptor file as either",basefilename+".properties or",filename)
            return []
    byteoffset = 0
    # used to store fields, their type, etc
    fieldnames = []
    fieldtype = []
    fieldindex = []

    if (ibinary == 0):
        # load ascii file
        halofile = open(filename, 'r')
        # read header information
        [filenum, numfiles] = halofile.readline().split()
        filenum = int(filenum)
        numfiles = int(numfiles)
        [numhalos, numtothalos] = halofile.readline().split()
        numhalos = np.uint64(numhalos)
        numtothalos = np.uint64(numtothalos)
        names = ((halofile.readline())).split()
        # remove the brackets in ascii file names
        fieldnames = [fieldname.split("(")[0] for fieldname in names]
        for i in np.arange(fieldnames.__len__()):
            fieldname = fieldnames[i]
            if fieldname in ["ID", "numSubStruct", "npart", "n_gas", "n_star", "Structuretype"]:
                fieldtype.append(np.uint64)
            elif fieldname in ["ID_mbp", "hostHaloID"]:
                fieldtype.append(np.int64)
            else:
                fieldtype.append(np.float64)
        halofile.close()
        # if desiredfields is NULL load all fields
        # but if this is passed load only those fields
        if (len(desiredfields) > 0):
            lend = len(desiredfields)
            fieldindex = np.zeros(lend, dtype=int)
            desiredfieldtype = [[] for i in range(lend)]
            for i in range(lend):
                fieldindex[i] = fieldnames.index(desiredfields[i])
                desiredfieldtype[i] = fieldtype[fieldindex[i]]
            fieldtype = desiredfieldtype
            fieldnames = desiredfields
        # to store the string containing data format
        fieldtypestring = ''
        for i in np.arange(fieldnames.__len__()):
            if fieldtype[i] == np.uint64:
                fieldtypestring += 'u8, '
            elif fieldtype[i] == np.int64:
                fieldtypestring += 'i8, '
            elif fieldtype[i] == np.float64:
                fieldtypestring += 'f8, '

    elif (ibinary == 1):
        # load binary file
        halofile = open(filename, 'rb')
        [filenum, numfiles] = np.fromfile(halofile, dtype=np.int32, count=2)
        [numhalos, numtothalos] = np.fromfile(
            halofile, dtype=np.uint64, count=2)
        headersize = np.fromfile(halofile, dtype=np.int32, count=1)[0]
        byteoffset = np.dtype(np.int32).itemsize*3 + \
            np.dtype(np.uint64).itemsize*2+4*headersize
        for i in range(headersize):
            fieldnames.append(struct.unpack('s', halofile.read(CHARSIZE)).strip())
        for i in np.arange(fieldnames.__len__()):
            fieldname = fieldnames[i]
            if fieldname in ["ID", "numSubStruct", "npart", "n_gas", "n_star", "Structuretype"]:
                fieldtype.append(np.uint64)
            elif fieldname in ["ID_mbp", "hostHaloID"]:
                fieldtype.append(np.int64)
            else:
                fieldtype.append(np.float64)
        halofile.close()
        # if desiredfields is NULL load all fields
        # but if this is passed load only those fields
        if (len(desiredfields) > 0):
            lend = len(desiredfields)
            fieldindex = np.zeros(lend, dtype=int)
            desiredfieldtype = [[] for i in range(lend)]
            for i in range(lend):
                fieldindex[i] = fieldnames.index(desiredfields[i])
                desiredfieldtype[i] = fieldtype[fieldindex[i]]
            fieldtype = desiredfieldtype
            fieldnames = desiredfields
        # to store the string containing data format
        fieldtypestring = ''
        for i in np.arange(fieldnames.__len__()):
            if fieldtype[i] == np.uint64:
                fieldtypestring += 'u8, '
            elif fieldtype[i] == np.int64:
                fieldtypestring += 'i8, '
            elif fieldtype[i] == np.float64:
                fieldtypestring += 'f8, '

    elif (ibinary == 2):
        # load hdf file
        halofile = h5py.File(filename, 'r')
        filenum = int(halofile["File_id"][0])
        numfiles = int(halofile["Num_of_files"][0])
        numhalos = np.uint64(halofile["Num_of_groups"][0])
        numtothalos = np.uint64(halofile["Total_num_of_groups"][0])
        #atime = np.float(halofile.attrs["Time"])
        fieldnames = [str(n) for n in halofile.keys()]
        # clean of header info
        fieldnames.remove("File_id")
        fieldnames.remove("Num_of_files")
        fieldnames.remove("Num_of_groups")
        fieldnames.remove("Total_num_of_groups")
        if ('Configuration' in fieldnames): fieldnames.remove('Configuration')
        if ('SimulationInfo' in fieldnames): fieldnames.remove('SimulationInfo')
        if ('UnitInfo' in fieldnames): fieldnames.remove('UnitInfo')
        fieldtype = [halofile[fieldname].dtype for fieldname in fieldnames]
        # if the desiredfields argument is passed only these fieds are loaded
        if (len(desiredfields) > 0):
            if (iverbose):
                print("Loading subset of all fields in property file ",
                      len(desiredfields), " instead of ", len(fieldnames))
            fieldnames = desiredfields
            fieldtype = [halofile[fieldname].dtype for fieldname in fieldnames]
        halofile.close()

    # allocate memory that will store the halo dictionary
    catalog = {fieldnames[i]: np.zeros(
        numtothalos, dtype=fieldtype[i]) for i in range(len(fieldnames))}
    noffset = np.uint64(0)
    for ifile in range(numfiles):
        if (inompi == True):
            filename = basefilename+".properties"
        else:
            filename = basefilename+".properties"+"."+str(ifile)
        if (iverbose):
            print("reading ", filename)
        if (ibinary == 0):
            halofile = open(filename, 'r')
            halofile.readline()
            numhalos = np.uint64(halofile.readline().split()[0])
            halofile.close()
            if (numhalos > 0):
                htemp = np.loadtxt(filename, skiprows=3, usecols=fieldindex,
                                   dtype=fieldtypestring, unpack=True, ndmin=1)
        elif(ibinary == 1):
            halofile = open(filename, 'rb')
            np.fromfile(halofile, dtype=np.int32, count=2)
            numhalos = np.fromfile(halofile, dtype=np.uint64, count=2)[0]
            # halofile.seek(byteoffset);
            if (numhalos > 0):
                htemp = np.fromfile(
                    halofile, usecols=fieldindex, dtype=fieldtypestring, unpack=True)
            halofile.close()
        elif(ibinary == 2):
            # here convert the hdf information into a numpy array
            halofile = h5py.File(filename, 'r')
            numhalos = np.uint64(halofile["Num_of_groups"][0])
            if (numhalos > 0):
                htemp = [np.array(halofile[catvalue])
                         for catvalue in fieldnames]
            halofile.close()
        #numhalos = len(htemp[0])
        for i in range(len(fieldnames)):
            catvalue = fieldnames[i]
            if (numhalos > 0):
                catalog[catvalue][noffset:noffset+numhalos] = htemp[i]
        noffset += numhalos
    # if subhalos are written in separate files, then read them too
    if (iseparatesubfiles == 1):
        for ifile in range(numfiles):
            if (inompi == True):
                filename = basefilename+".sublevels"+".properties"
            else:
                filename = basefilename+".sublevels" + \
                    ".properties"+"."+str(ifile)
            if (iverbose):
                print("reading ", filename)
            if (ibinary == 0):
                halofile = open(filename, 'r')
                halofile.readline()
                numhalos = np.uint64(halofile.readline().split()[0])
                halofile.close()
                if (numhalos > 0):
                    htemp = np.loadtxt(
                        filename, skiprows=3, usecols=fieldindex, dtype=fieldtypestring, unpack=True, ndmin=1)
            elif(ibinary == 1):
                halofile = open(filename, 'rb')
                # halofile.seek(byteoffset);
                np.fromfile(halofile, dtype=np.int32, count=2)
                numhalos = np.fromfile(halofile, dtype=np.uint64, count=2)[0]
                if (numhalos > 0):
                    htemp = np.fromfile(
                        halofile, usecols=fieldindex, dtype=fieldtypestring, unpack=True)
                halofile.close()
            elif(ibinary == 2):
                halofile = h5py.File(filename, 'r')
                numhalos = np.uint64(halofile["Num_of_groups"][0])
                if (numhalos > 0):
                    htemp = [np.array(halofile[catvalue])
                             for catvalue in fieldnames]
                halofile.close()
            #numhalos = len(htemp[0])
            for i in range(len(fieldnames)):
                catvalue = fieldnames[i]
            if (numhalos > 0):
                catalog[catvalue][noffset:noffset+numhalos] = htemp[i]
            noffset += numhalos
    # load associated simulation info, time, units and configuration options
    if (isiminfo):
        if(iverbose):
            print("reading ",basefilename+".siminfo")
        catalog['SimulationInfo'] = ReadSimInfo(basefilename)

    if (iunitinfo):
        if(iverbose):
            print("reading ",basefilename+".unitinfo")
        catalog['UnitInfo'] = ReadUnitInfo(basefilename)

    if (iconfiginfo):
        if(iverbose):
            print("reading ",basefilename+".configuration")
        catalog['ConfigurationInfo'] = ReadConfigInfo(basefilename)

    if (iverbose):
        print("done reading properties file ", time.process_time()-start)
    return catalog, numtothalos


def ReadPropertyFileMultiWrapper(basefilename, index, halodata, numhalos, atime, ibinary=0, iseparatesubfiles=0, iverbose=0, desiredfields=[]):
    """
    Wrapper for multithreaded reading
    """
    # call read routine and store the data
    halodata[index], numhalos[index] = ReadPropertyFile(
        basefilename, ibinary, iseparatesubfiles, iverbose, desiredfields)


def ReadPropertyFileMultiWrapperNamespace(index, basefilename, ns, ibinary=0, iseparatesubfiles=0, iverbose=0, desiredfields=[]):
    # call read routine and store the data
    ns.hdata[index], ns.ndata[index] = ReadPropertyFile(
        basefilename, ibinary, iseparatesubfiles, iverbose, desiredfields)


def ReadHaloMergerTree(treefilename, ibinary=0, iverbose=0, imerit=False, inpart=False):
    """
    VELOCIraptor/STF merger tree in ascii format contains
    a header with
        number_of_snapshots
        a description of how the tree was built
        total number of halos across all snapshots

    then followed by data
    for each snapshot
        snapshotvalue numhalos
        haloid_1 numprogen_1
        progenid_1
        progenid_2
        ...
        progenid_numprogen_1
        haloid_2 numprogen_2
        .
        .
        .
    one can also have an output format that has an additional field for each progenitor, the meritvalue

    """
    start = time.process_time()
    tree = {}
    if (iverbose):
        print("reading Tree file", treefilename, os.path.isfile(treefilename))
    if (os.path.isfile(treefilename) == False):
        print("Error, file not found")
        return tree
    # if ascii format
    if (ibinary == 0):
        treefile = open(treefilename, 'r')
        numsnap = int(treefile.readline())
        treefile.close()
    elif(ibinary == 2):
        if (iverbose):
            print("Reading HDF5 input")

        snaptreelist = open(treefilename, 'r')
        snaptreename = snaptreelist.readline().strip()+".tree"
        numsnaplist = sum(1 for line in snaptreelist) +1

        treedata = h5py.File(snaptreename, "r")
        numsnap = treedata.attrs['Number_of_snapshots']

        #Check if the treefrog number of snapshots and the number of files in the list is consistent
        if(numsnap!=numsnaplist):
            print("Error, the number of snapshots reported by the TreeFrog output is different to the number of filenames supplied. \nPlease update this.")
            return tree

        #Lets extract te header information
        tree["Header"] = dict()
        for field in treedata.attrs.keys():
            tree["Header"][field] = treedata.attrs[field]

        treedata.close()
        snaptreelist.close()
    else:
        print("Unknown format, returning null")
        numsnap = 0
        return tree

    tree.update({i: {"haloID": [], "Num_progen": [], "Progen": []}
            for i in range(numsnap)})
    if (imerit):
        for i in range(numsnap):
            tree[i]['Merit'] = []
    if (inpart):
        for i in range(numsnap):
            tree[i]['Npart'] = []
            tree[i]['Npart_progen'] = []

    # if ascii format
    if (ibinary == 0):
        treefile = open(treefilename, 'r')
        numsnap = int(treefile.readline())
        descrip = treefile.readline().strip()
        tothalos = int(treefile.readline())
        offset = 0
        totalnumprogen = 0
        for i in range(numsnap):
            [snapval, numhalos] = treefile.readline().strip().split('\t')
            snapval = int(snapval)
            numhalos = int(numhalos)
            # if really verbose
            if (iverbose == 2):
                print(snapval, numhalos)
            tree[i]["haloID"] = np.zeros(numhalos, dtype=np.int64)
            tree[i]["Num_progen"] = np.zeros(numhalos, dtype=np.uint32)
            tree[i]["Progen"] = [[] for j in range(numhalos)]
            if (imerit):
                tree[i]["Merit"] = [[] for j in range(numhalos)]
            if (inpart):
                tree[i]["Npart"] = np.zeros(numhalos, dtype=np.uint32)
                tree[i]["Npart_progen"] = [[] for j in range(numhalos)]
            for j in range(numhalos):
                data = treefile.readline().strip().split('\t')
                hid = np.int64(data[0])
                nprog = np.uint32(data[1])
                tree[i]["haloID"][j] = hid
                tree[i]["Num_progen"][j] = nprog
                if (inpart):
                    tree[i]["Npart"][j] = np.uint32(data[2])
                totalnumprogen += nprog
                if (nprog > 0):
                    tree[i]["Progen"][j] = np.zeros(nprog, dtype=np.int64)
                    if (imerit):
                        tree[i]["Merit"][j] = np.zeros(nprog, dtype=np.float32)
                    if (inpart):
                        tree[i]["Npart_progen"][j] = np.zeros(
                            nprog, dtype=np.uint32)
                    for k in range(nprog):
                        data = treefile.readline().strip().split(' ')
                        tree[i]["Progen"][j][k] = np.int64(data[0])
                        if (imerit):
                            tree[i]["Merit"][j][k] = np.float32(data[1])
                        if (inpart):
                            tree[i]["Npart_progen"][j][k] = np.uint32(data[2])

    elif(ibinary == 2):

        if("HaloID_snapshot_offset" in tree["Header"]):
            snapshotoffset = tree["Header"]["HaloID_snapshot_offset"]
        else:
            print("Warning: you are using older TreeFrog output (version<=1.2) which does not contain information about which snapshot the halo catalog starts at\nAssuming that it starts at snapshot = 0.\nPlease use a TreeFrog version>1.2 if you require this feature")
            snapshotoffset = 0

        snaptreelist = open(treefilename, 'r')
        for snap in range(snapshotoffset,snapshotoffset+numsnap):
            snaptreename = snaptreelist.readline().strip()+".tree"
            if (iverbose):
                print("Reading", snaptreename)
            treedata = h5py.File(snaptreename, "r")

            tree[snap]["haloID"] = np.asarray(treedata["ID"])
            tree[snap]["Num_progen"] = np.asarray(treedata["NumProgen"])
            if(inpart):
                tree[snap]["Npart"] = np.asarray(treedata["Npart"])

            # See if the dataset exits
            if("ProgenOffsets" in treedata.keys()):

                # Find the indices to split the array
                split = np.add(np.asarray(
                    treedata["ProgenOffsets"]), tree[snap]["Num_progen"], dtype=np.uint64, casting="unsafe")

                # Read in the progenitors, splitting them as reading them in
                tree[snap]["Progen"] = np.split(
                    treedata["Progenitors"][:], split[:-1])

                if(inpart):
                    tree[snap]["Npart_progen"] = np.split(
                        treedata["ProgenNpart"], split[:-1])
                if(imerit):
                    tree[snap]["Merit"] = np.split(
                        treedata["Merits"], split[:-1])

        snaptreelist.close()
    if (iverbose):
        print("done reading tree file ", time.process_time()-start)
    return tree

class MinStorageList():
    """
    This uses two smaller arrays to access a large contiguous array and return subchunks
    """
    Num,Offset=None,None
    Data=None
    def __init__(self,nums,offsets,rawdata):
        self.Num=nums
        self.Offset=offsets
        self.Data=rawdata
    def __getitem__(self, index):
        if self.Num[index] == 0 :
            return np.array([])
        else :
            return self.Data[self.Offset[index]:self.Offset[index]+self.Num[index]]
    def GetBestRanks(self, ):
        return np.array(self.Data[self.Offset],copy=True)


def ReadHaloMergerTreeDescendant(treefilename, ireverseorder=False, ibinary=0,
                                 iverbose=0, imerit=False, inpart=False,
                                 ireducedtobestranks=False, meritlimit=0.025,
                                 ireducemem=True,iprimarydescen=False):
    """
    VELOCIraptor/STF descendant based merger tree in ascii format contains
    a header with
        number_of_snapshots
        a description of how the tree was built
        total number of halos across all snapshots

    then followed by data
    for each snapshot
        snapshotvalue numhalos
        haloid_1 numprogen_1
        progenid_1
        progenid_2
        ...
        progenid_numprogen_1
        haloid_2 numprogen_2
        .
        .
        .
    one can also have an output format that has an additional field for each progenitor, the meritvalue

    """
    start = time.process_time()
    tree = {}
    if (iverbose):
        print("reading Tree file", treefilename, os.path.isfile(treefilename))
    if (os.path.isfile(treefilename) == False):
        print("Error, file not found")
        return tree
    # fine out how many snapshots there are
    # if ascii format
    if (ibinary == 0):
        if (iverbose):
            print("Reading ascii input")
        treefile = open(treefilename, 'r')
        numsnap = int(treefile.readline())
        treefile.close()
    # hdf format, input file is a list of filenames can also extract the header info
    elif(ibinary == 2):
        if (iverbose):
            print("Reading HDF5 input")

        snaptreelist = open(treefilename, 'r')
        snaptreename = snaptreelist.readline().strip()+".tree"
        numsnap = sum(1 for line in snaptreelist) +1

        treedata = h5py.File(snaptreename, "r")

        #Lets extract te header information
        tree["Header"] = dict()
        for field in treedata.attrs.keys():
            tree["Header"][field] = treedata.attrs[field]

        treedata.close()
        snaptreelist.close()
    else:
        print("Unknown format, returning null")
        numsnap = 0
        return tree

    #Check for flag compatibility
    if(ireducemem & iprimarydescen):
        print("Warning: Both the ireducemem and iprimarydescen are set to True. The ireducemem is not needed for iprimarydescen flag \nas it already had reduced memory footprint, due to only primiary descendants being extracted")
        ireducemem=False

    #Update the dictionary with the tree information
    tree.update({i: {"haloID": [], "Num_descen": [], "Descen": [], "Rank": []}
            for i in range(numsnap)})
    if (imerit):
        for i in range(numsnap):
            tree[i]['Merit'] = []
    if (inpart):
        for i in range(numsnap):
            tree[i]['Npart'] = []
            tree[i]['Npart_descen'] = []

    if (ibinary == 0):
        treefile = open(treefilename, 'r')
        numsnap = int(treefile.readline())
        descrip = treefile.readline().strip()
        tothalos = int(treefile.readline())
        offset = 0
        totalnumdescen = 0
        for i in range(numsnap):
            ii = i
            if (ireverseorder):
                ii = numsnap-1-i
            [snapval, numhalos] = treefile.readline().strip().split('\t')
            snapval = int(snapval)
            numhalos = int(numhalos)
            # if really verbose
            if (iverbose == 2):
                print(snapval, numhalos)
            tree[ii]["haloID"] = np.zeros(numhalos, dtype=np.int64)
            tree[ii]["Num_descen"] = np.zeros(numhalos, dtype=np.uint32)
            tree[ii]["Descen"] = [[] for j in range(numhalos)]
            tree[ii]["Rank"] = [[] for j in range(numhalos)]
            if (imerit):
                tree[ii]["Merit"] = [[] for j in range(numhalos)]
            if (inpart):
                tree[i]["Npart"] = np.zeros(numhalos, dtype=np.uint32)
                tree[ii]["Npart_descen"] = [[] for j in range(numhalos)]
            for j in range(numhalos):
                data = treefile.readline().strip().split('\t')
                hid = np.int64(data[0])
                ndescen = np.uint32(data[1])
                tree[ii]["haloID"][j] = hid
                tree[ii]["Num_descen"][j] = ndescen
                if (inpart):
                    tree[ii]["Npart"][j] = np.uint32(data[2])
                totalnumdescen += ndescen
                if (ndescen > 0):
                    tree[ii]["Descen"][j] = np.zeros(ndescen, dtype=np.int64)
                    tree[ii]["Rank"][j] = np.zeros(ndescen, dtype=np.uint32)
                    if (imerit):
                        tree[ii]["Merit"][j] = np.zeros(
                            ndescen, dtype=np.float32)
                    if (inpart):
                        tree[ii]["Npart_descen"][j] = np.zeros(
                            ndescen, dtype=np.float32)
                    for k in range(ndescen):
                        data = treefile.readline().strip().split(' ')
                        tree[ii]["Descen"][j][k] = np.int64(data[0])
                        tree[ii]["Rank"][j][k] = np.uint32(data[1])
                        if (imerit):
                            tree[ii]["Merit"][j][k] = np.float32(data[2])
                        if (inpart):
                            tree[ii]["Npart_descen"][j][k] = np.uint32(data[3])

                if (ireducedtobestranks):
                    halolist=np.where(tree[ii]["Num_descen"]>1)[0]
                    for ihalo in halolist:
                        numdescen = 1
                        if (imerit):
                            numdescen = np.int32(np.max([1,np.argmax(tree[ii]["Merit"][ihalo]<meritlimit)]))
                        tree[ii]["Num_descen"][ihalo] = numdescen
                        tree[ii]["Descen"][ihalo] = np.array([tree[ii]["Descen"][ihalo][:numdescen]])
                        tree[ii]["Rank"][ihalo] = np.array([tree[ii]["Rank"][ihalo][:numdescen]])
                        if (imerit):
                            tree[ii]["Merit"][ihalo] = np.array([tree[ii]["Merit"][ihalo][:numdescen]])
                        if (inpart):
                            tree[ii]["Npart_descen"][ihalo] = np.array([tree[ii]["Npart_descen"][ihalo][:numdescen]])
    # hdf format
    elif(ibinary == 2):

        snaptreelist = open(treefilename, 'r')
        snaptreenames=[[] for snap in range(numsnap)]
        for snap in range(numsnap):
            snaptreenames[snap] = snaptreelist.readline().strip()+".tree"
        snaptreelist.close()

        if("HaloID_snapshot_offset" in tree["Header"]):
            snapshotoffset = tree["Header"]["HaloID_snapshot_offset"]
        else:
            print("Warning: you are using older TreeFrog output (version<=1.2) which does not contain information about which snapshot the halo catalog starts at\nAssuming that it starts at snapshot = 0.\nPlease use a TreeFrog version>1.2 if you require this feature")
            snapshotoffset = 0

        if (iverbose):
            snaptreename = snaptreenames[0]
            treedata = h5py.File(snaptreename, "r")
            numhalos = treedata.attrs["Total_number_of_halos"]
            memsize = np.uint64(1).nbytes*2.0+np.uint16(0).nbytes*2.0
            if (inpart):
                memsize += np.int32(0).nbytes*2.0
            if (imerit):
                memsize += np.float32(0).nbytes
            memsize *= numhalos
            if (ireducemem):
                print("Reducing memory, changes api.")
                print("Data contains ", numhalos, "halos and will likley need a minimum of", memsize/1024.**3.0, "GB of memory")
            elif(iprimarydescen):
                print("Extracting just the primary descendants")
                print("Data contains ", numhalos, "halos and will likley need a minimum of", memsize/1024.**3.0, "GB of memory")
            else:
                print("Contains ", numhalos, "halos and will likley minimum ", memsize/1024.**3.0, "GB of memory")
                print("Plus overhead to store list of arrays, with likely need a minimum of ",100*numhalos/1024**3.0, "GB of memory ")
            treedata.close()
        for snap in range(snapshotoffset,snapshotoffset+numsnap):
            snaptreename = snaptreenames[snap]

            if (iverbose):
                print("Reading", snaptreename)
            treedata = h5py.File(snaptreename, "r")
            tree[snap]["haloID"] = np.asarray(treedata["ID"])
            tree[snap]["Num_descen"] = np.asarray(treedata["NumDesc"],np.uint16)
            numhalos=tree[snap]["haloID"].size
            if(inpart):
                tree[snap]["Npart"] = np.asarray(treedata["Npart"],np.int32)

            # See if the dataset exits
            if("DescOffsets" in treedata.keys()):

                # Find the indices to split the array
                if (ireducemem):
                    tree[snap]["_Offsets"] = np.asarray(treedata["DescOffsets"],dtype=np.uint64)
                elif(iprimarydescen):
                    offsets = np.asarray(treedata["DescOffsets"],dtype=np.uint64)
                    #Only include the offset for the last halo in the array if it has a descendant
                    if(offsets.size):
                        if(tree[snap]["Num_descen"][-1]==0): offsets = offsets[:-1]
                else:
                    descenoff=np.asarray(treedata["DescOffsets"],dtype=np.uint64)
                    split = np.add(descenoff, tree[snap]["Num_descen"], dtype=np.uint64, casting="unsafe")[:-1]
                    descenoff=None

                # Read in the data splitting it up as reading it in
                # if reducing memory then store all the values in the _ keys
                # and generate class that returns the appropriate subchunk as an array when using the [] operaotor
                # otherwise generate lists of arrays
                if (ireducemem):
                    tree[snap]["_Ranks"] = np.asarray(treedata["Ranks"],dtype=np.int16)
                    tree[snap]["_Descens"] = np.asarray(treedata["Descendants"],dtype=np.uint64)
                    tree[snap]["Rank"] = MinStorageList(tree[snap]["Num_descen"],tree[snap]["_Offsets"],tree[snap]["_Ranks"])
                    tree[snap]["Descen"] = MinStorageList(tree[snap]["Num_descen"],tree[snap]["_Offsets"],tree[snap]["_Descens"])
                elif(iprimarydescen):
                    tree[snap]["Rank"] = np.asarray(treedata["Ranks"],dtype=np.uint16)[offsets]
                    tree[snap]["Descen"] = np.asarray(treedata["Descendants"],dtype=np.uint64)[offsets]
                else:
                    tree[snap]["Rank"] = np.split(np.asarray(treedata["Ranks"],dtype=np.uint16), split)
                    tree[snap]["Descen"] = np.split(np.asarray(treedata["Descendants"],dtype=np.uint64), split)

                if(inpart):
                    if (ireducemem):
                        tree[snap]["_Npart_descens"] = np.asarray(treedata["DescenNpart"],np.uint64)
                        tree[snap]["Npart_descen"] = MinStorageList(tree[snap]["Num_descen"],tree[snap]["_Offsets"],tree[snap]["_Npart_descens"])
                    elif(iprimarydescen):
                        tree[snap]["Npart_descen"] = np.asarray(treedata["DescenNpart"],np.uint64)[offsets]
                    else:
                        tree[snap]["Npart_descen"] = np.split(np.asarray(treedata["DescenNpart"],np.uint64), split)
                if(imerit):
                    if (ireducemem):
                        tree[snap]["_Merits"] = np.asarray(treedata["Merits"],np.float32)
                        tree[snap]["Merit"] = MinStorageList(tree[snap]["Num_descen"],tree[snap]["_Offsets"],tree[snap]["_Merits"])
                    elif(iprimarydescen):
                        tree[snap]["Merit"] = np.asarray(treedata["Merits"],np.float32)[offsets]
                    else:
                        tree[snap]["Merit"] = np.split(np.asarray(treedata["Merits"],np.float32), split)
                #if reducing stuff down to best ranks, then only keep first descendant
                #unless also reading merit and then keep first descendant and all other descendants that are above a merit limit
                if (ireducedtobestranks==True and ireducemem==False and iprimarydescen==False):
                    halolist = np.where(tree[snap]["Num_descen"]>1)[0]
                    if (iverbose):
                        print('Reducing memory needed. At snap ', snap, ' with %d total halos and alter %d halos. '% (len(tree[snap]['Num_descen']), len(halolist)))
                        print(np.percentile(tree[snap]['Num_descen'][halolist],[50.0,99.0]))
                    for ihalo in halolist:
                        numdescen = 1
                        if (imerit):
                            numdescen = np.int32(np.max([1,np.argmax(tree[snap]["Merit"][ihalo]<meritlimit)]))
                        tree[snap]["Num_descen"][ihalo] = numdescen
                        tree[snap]["Descen"][ihalo] = np.asarray([tree[snap]["Descen"][ihalo][:numdescen]])
                        tree[snap]["Rank"][ihalo] = np.asarray([tree[snap]["Rank"][ihalo][:numdescen]])
                        if (imerit):
                            tree[snap]["Merit"][ihalo] = np.asarray([tree[snap]["Merit"][ihalo][:numdescen]])
                        if (inpart):
                            tree[snap]["Npart_descen"][ihalo] = np.asarray([tree[snap]["Npart_descen"][ihalo][:numdescen]])
                split=None
            treedata.close()

    if (iverbose):
        print("done reading tree file ", time.process_time()-start)
    return tree


def ReadHaloPropertiesAcrossSnapshots(numsnaps, snaplistfname, inputtype, iseperatefiles, iverbose=0, desiredfields=[]):
    """
    read halo data from snapshots listed in file with snaplistfname file name
    """
    halodata = [dict() for j in range(numsnaps)]
    ngtot = [0 for j in range(numsnaps)]
    atime = [0 for j in range(numsnaps)]
    start = time.process_time()
    print("reading data")
    # if there are a large number of snapshots to read, read in parallel
    # only read in parallel if worthwhile, specifically if large number of snapshots and snapshots are ascii
    iparallel = (numsnaps > 20 and inputtype == 2)
    if (iparallel):
        # determine maximum number of threads
        nthreads = min(mp.cpu_count(), numsnaps)
        nchunks = int(np.ceil(numsnaps/float(nthreads)))
        print("Using", nthreads, "threads to parse ",
              numsnaps, " snapshots in ", nchunks, "chunks")
        # load file names
        snapnamelist = open(snaplistfname, 'r')
        catfilename = ["" for j in range(numsnaps)]
        for j in range(numsnaps):
            catfilename[j] = snapnamelist.readline().strip()
        # allocate a manager
        manager = mp.Manager()
        # use manager to specify the dictionary and list that can be accessed by threads
        hdata = manager.list([manager.dict() for j in range(numsnaps)])
        ndata = manager.list([0 for j in range(numsnaps)])
        adata = manager.list([0 for j in range(numsnaps)])
        # now for each chunk run a set of proceses
        for j in range(nchunks):
            offset = j*nthreads
            # if last chunk then must adjust nthreads
            if (j == nchunks-1):
                nthreads = numsnaps-offset
            # when calling a process pass manager based proxies, which then are used to copy data back
            processes = [mp.Process(target=ReadPropertyFileMultiWrapper, args=(catfilename[offset+k], k+offset,
                                                                               hdata, ndata, adata, inputtype, iseperatefiles, iverbose, desiredfields)) for k in range(nthreads)]
            # start each process
            # store the state of each thread, alive or not, and whether it has finished
            activethreads = [[True, False] for k in range(nthreads)]
            count = 0
            for p in processes:
                print("reading", catfilename[offset+count])
                p.start()
                # space threads apart (join's time out is 0.25 seconds
                p.join(0.2)
                count += 1
            totactivethreads = nthreads
            while(totactivethreads > 0):
                count = 0
                for p in processes:
                    # join thread and see if still active
                    p.join(0.5)
                    if (p.is_alive() == False):
                        # if thread nolonger active check if its been processed
                        if (activethreads[count][1] == False):
                            # make deep copy of manager constructed objects that store data
                            #halodata[i][offset+count] = copy.deepcopy(hdata[offset+count])
                            # try instead init a dictionary
                            halodata[offset+count] = dict(hdata[offset+count])
                            ngtot[offset+count] = ndata[offset+count]
                            atime[offset+count] = adata[offset+count]
                            # effectively free the data in manager dictionary
                            hdata[offset+count] = []
                            activethreads[count][0] = False
                            activethreads[count][1] = True
                            totactivethreads -= 1
                    count += 1
            # terminate threads
            for p in processes:
                p.terminate()

    else:
        snapnamelist = open(snaplistfname, 'r')
        for j in range(0, numsnaps):
            catfilename = snapnamelist.readline().strip()
            print("reading ", catfilename)
            halodata[j], ngtot[j], atime[j] = ReadPropertyFile(
                catfilename, inputtype, iseperatefiles, iverbose, desiredfields)
    print("data read in ", time.process_time()-start)
    return halodata, ngtot, atime


def ReadCrossCatalogList(fname, meritlim=0.1, iverbose=0):
    """
    Reads a cross catalog produced by halomergertree,
    also allows trimming of cross catalog using a higher merit threshold than one used to produce catalog
    """
    return []
    """
    start = time.process_time()
    if (iverbose):
        print("reading cross catalog")
    dfile = open(fname, "r")
    dfile.readline()
    dfile.readline()
    dataline = (dfile.readline().strip()).split('\t')
    ndata = np.int32(dataline[1])
    pdata = CrossCatalogList(ndata)
    for i in range(0, ndata):
        data = (dfile.readline().strip()).split('\t')
        nmatches = np.int32(data[1])
        for j in range(0, nmatches):
            data = (dfile.readline().strip()).split(' ')
            meritval = np.float32(data[1])
            nsharedval = np.float32(data[2])
            if(meritval > meritlim):
                nmatchid = np.int64(data[0])
                pdata.matches[i].append(nmatchid)
                pdata.matches[i].append(meritval)
                pdata.nsharedfrac[i].append(nsharedval)
                pdata.nmatches[i] += 1
    dfile.close()
    if (iverbose):
        print("done reading cross catalog ", time.process_time()-start)
    return pdata
    """


def ReadSimInfo(basefilename):
    """
    Reads in the information in the .siminfo file and returns it as a dictionary
    """

    filename = basefilename + ".siminfo"

    if (os.path.isfile(filename) == False):
        print(filename,"not found")
        return {}

    cosmodata = {}
    siminfofile = open(filename, "r")
    for i,l in enumerate(siminfofile):

        line = l.strip()

        #See if this a comment
        if(line==""):
            continue

        if(line[0]=="#"):
            continue

        try:
            field, value = line.replace(" ","").split(':')
        except ValueError:
            print("Cannot read line",i,"of",filename,"continuing")
            continue

        #See if the datatype is present
        if("#" in value):
            value, datatype = value.split("#")
            try:
                typefunc = np.dtype(datatype).type
            except:
                print("Cannot interpret",datatype,"for field",field,"on line",i,"in",filename,"as a valid datatype, interpreting the value as float64")
                typefunc = np.float64
        else:
            typefunc = np.float64

        try:
            #Find if the value is a list of values
            if("," in value):
                value = value.split(",")

                #Remove any empty strings
                value = list(filter(None,value))
                cosmodata[field] = np.array(value,dtype=typefunc)
            else:
                cosmodata[field] = typefunc(value)
        except ValueError:
            print("Cannot interpret",value,"as a",np.dtype(typefunc))

    siminfofile.close()
    return cosmodata


def ReadUnitInfo(basefilename):
    """
    Reads in the information in the .units file and returns it as a dictionary
    """

    filename = basefilename + ".units"

    if (os.path.isfile(filename) == False):
        print(filename,"not found")
        return {}

    unitsfile = open(filename, 'r')
    unitdata = dict()
    for i,l in enumerate(unitsfile):

        line = l.strip()

        #See if this a comment
        if(line==""):
            continue

        if(line[0]=="#"):
            continue

        try:
            field, value = line.replace(" ","").split(':')
        except ValueError:
            print("Cannot read line",i,"of",filename,"continuing")
            continue

        #See if the datatype is present
        if("#" in value):
            value, datatype = value.split("#")
            try:
                typefunc = np.dtype(datatype).type
            except:
                print("Cannot interpret",datatype,"for field",field,"on line",i,"in",filename,"as a valid datatype, interpreting the value as float64")
                typefunc = np.float64
        else:
            typefunc = np.float64

        try:
            #Find if the value is a list of values
            if("," in value):
                value = value.split(",")

                #Remove any empty strings
                value = list(filter(None,value))
                unitdata[field] = np.array(value,dtype=typefunc)
            else:
                unitdata[field] = typefunc(value)
        except ValueError:
            print("Cannot interpret",value,"as a",np.dtype(typefunc))

    unitsfile.close()
    return unitdata


def ReadConfigInfo(basefilename):
    """
    Reads in the information in the .configuration file and returns it as a dictionary
    """
    filename = basefilename+".configuration"

    if (os.path.isfile(filename) == False):
        print(filename,"not found")
        return {}

    configfile = open(filename, 'r')
    configdata = dict()
    for i,l in enumerate(configfile):

        line = l.strip()

        #See if this a comment
        if(line==""):
            continue

        if(line[0]=="#"):
            continue

        try:
            field, value = line.replace(" ","").split(':')
        except ValueError:
            try:
                field, value = line.replace(" ","").split('=')
            except ValueError:
                print("Cannot read line",i,"of",filename,"continuing")
                continue

        #See if the datatype is present
        if("#" in value):
            value, datatype = value.split("#")
            try:
                typefunc = np.dtype(datatype).type
            except:
                print("Cannot interpret",datatype,"for field",field,"on line",i,"in",filename,"as a valid datatype, interpreting the value as float64")
                typefunc = np.float64
        else:
            typefunc = np.float64

        try:
            #Find if the value is a list of values
            if("," in value):
                value = value.split(",")

                #Remove any empty strings
                value = list(filter(None,value))
                configdata[field] = np.array(value,dtype=typefunc)
            else:
                configdata[field] = typefunc(value)
        except ValueError:
            print("Cannot interpret",value,"as a",np.dtype(typefunc))

    configfile.close()
    return configdata


def ReadParticleDataFile(basefilename, ibinary=0, iseparatesubfiles=0, iparttypes=0, iverbose=0, binarydtype=np.int64):
    """
    VELOCIraptor/STF catalog_group, catalog_particles and catalog_parttypes in various formats

    Note that a file will indicate how many files the total output has been split into

    """
    inompi = True
    if (iverbose):
        print("reading particle data", basefilename)
    gfilename = basefilename+".catalog_groups"
    pfilename = basefilename+".catalog_particles"
    upfilename = pfilename+".unbound"
    tfilename = basefilename+".catalog_parttypes"
    utfilename = tfilename+".unbound"
    # check for file existence
    if (os.path.isfile(gfilename) == True):
        numfiles = 0
    else:
        gfilename += ".0"
        pfilename += ".0"
        upfilename += ".0"
        tfilename += ".0"
        utfilename += ".0"
        inompi = False
        if (os.path.isfile(gfilename) == False):
            print("file not found")
            return []
    byteoffset = 0

    # load header information from file to get total number of groups
    # ascii
    if (ibinary == 0):
        gfile = open(gfilename, 'r')
        [filenum, numfiles] = gfile.readline().split()
        filenum = int(filenum)
        numfiles = int(numfiles)
        [numhalos, numtothalos] = gfile.readline().split()
        numhalos = np.uint64(numhalos)
        numtothalos = np.uint64(numtothalos)
    # binary
    elif (ibinary == 1):
        gfile = open(gfilename, 'rb')
        [filenum, numfiles] = np.fromfile(gfile, dtype=np.int32, count=2)
        [numhalos, numtothalos] = np.fromfile(gfile, dtype=np.uint64, count=2)
    # hdf
    elif (ibinary == 2):
        gfile = h5py.File(gfilename, 'r')
        filenum = int(gfile["File_id"][0])
        numfiles = int(gfile["Num_of_files"][0])
        numhalos = np.uint64(gfile["Num_of_groups"][0])
        numtothalos = np.uint64(gfile["Total_num_of_groups"][0])
    gfile.close()

    particledata = dict()
    particledata['Npart'] = np.zeros(numtothalos, dtype=np.uint64)
    particledata['Npart_unbound'] = np.zeros(numtothalos, dtype=np.uint64)
    particledata['Particle_IDs'] = [[] for i in range(numtothalos)]
    if (iparttypes == 1):
        particledata['Particle_Types'] = [[] for i in range(numtothalos)]

    # now for all files
    counter = np.uint64(0)
    subfilenames = [""]
    if (iseparatesubfiles == 1):
        subfilenames = ["", ".sublevels"]
    for ifile in range(numfiles):
        for subname in subfilenames:
            bfname = basefilename+subname
            gfilename = bfname+".catalog_groups"
            pfilename = bfname+".catalog_particles"
            upfilename = pfilename+".unbound"
            tfilename = bfname+".catalog_parttypes"
            utfilename = tfilename+".unbound"
            if (inompi == False):
                gfilename += "."+str(ifile)
                pfilename += "."+str(ifile)
                upfilename += "."+str(ifile)
                tfilename += "."+str(ifile)
                utfilename += "."+str(ifile)
            if (iverbose):
                print("reading", bfname, ifile)

            # ascii
            if (ibinary == 0):
                gfile = open(gfilename, 'r')
                # read header information
                gfile.readline()
                [numhalos, foo] = gfile.readline().split()
                numhalos = np.uint64(numhalos)
                gfile.close()
                # load data
                gdata = np.loadtxt(gfilename, skiprows=2, dtype=np.uint64)
                numingroup = gdata[:numhalos]
                offset = gdata[int(numhalos):int(2*numhalos)]
                uoffset = gdata[int(2*numhalos):int(3*numhalos)]
                # particle id data
                pfile = open(pfilename, 'r')
                pfile.readline()
                [npart, foo] = pfile.readline().split()
                npart = np.uint64(npart)
                pfile.close()
                piddata = np.loadtxt(pfilename, skiprows=2, dtype=np.int64)
                upfile = open(upfilename, 'r')
                upfile.readline()
                [unpart, foo] = upfile.readline().split()
                unpart = np.uint64(unpart)
                upfile.close()
                upiddata = np.loadtxt(upfilename, skiprows=2, dtype=np.int64)
                if (iparttypes == 1):
                    # particle id data
                    tfile = open(tfilename, 'r')
                    tfile.readline()
                    [npart, foo] = tfile.readline().split()
                    tfile.close()
                    tdata = np.loadtxt(tfilename, skiprows=2, dtype=np.uint16)
                    utfile = open(utfilename, 'r')
                    utfile.readline()
                    [unpart, foo] = utfile.readline().split()
                    utfile.close()
                    utdata = np.loadtxt(
                        utfilename, skiprows=2, dtype=np.uint16)
            # binary
            elif (ibinary == 1):
                gfile = open(gfilename, 'rb')
                np.fromfile(gfile, dtype=np.int32, count=2)
                [numhalos, foo] = np.fromfile(gfile, dtype=np.uint64, count=2)
                # need to generalise to
                numingroup = np.fromfile(
                    gfile, dtype=binarydtype, count=numhalos)
                offset = np.fromfile(gfile, dtype=binarydtype, count=numhalos)
                uoffset = np.fromfile(gfile, dtype=binarydtype, count=numhalos)
                gfile.close()
                pfile = open(pfilename, 'rb')
                np.fromfile(pfile, dtype=np.int32, count=2)
                [npart, foo] = np.fromfile(pfile, dtype=np.uint64, count=2)
                piddata = np.fromfile(pfile, dtype=binarydtype, count=npart)
                pfile.close()
                upfile = open(upfilename, 'rb')
                np.fromfile(upfile, dtype=np.int32, count=2)
                [unpart, foo] = np.fromfile(upfile, dtype=np.uint64, count=2)
                upiddata = np.fromfile(upfile, dtype=binarydtype, count=unpart)
                upfile.close()
                if (iparttypes == 1):
                    tfile = open(tfilename, 'rb')
                    np.fromfile(tfile, dtype=np.int32, count=2)
                    [npart, foo] = np.fromfile(tfile, dtype=np.uint16, count=2)
                    tdata = np.fromfile(tfile, dtype=binarydtype, count=npart)
                    tfile.close()
                    utfile = open(utfilename, 'rb')
                    np.fromfile(utfile, dtype=np.int32, count=2)
                    [unpart, foo] = np.fromfile(
                        utfile, dtype=np.uint16, count=2)
                    utdata = np.fromfile(
                        utfile, dtype=binarydtype, count=unpart)
                    utfile.close()
            # hdf
            elif (ibinary == 2):
                gfile = h5py.File(gfilename, 'r')
                numhalos = np.uint64(gfile["Num_of_groups"][0])
                numingroup = np.uint64(gfile["Group_Size"])
                offset = np.uint64(gfile["Offset"])
                uoffset = np.uint64(gfile["Offset_unbound"])
                gfile.close()
                pfile = h5py.File(pfilename, 'r')
                upfile = h5py.File(upfilename, 'r')
                piddata = np.int64(pfile["Particle_IDs"])
                upiddata = np.int64(upfile["Particle_IDs"])
                npart = len(piddata)
                unpart = len(upiddata)

                pfile.close()
                upfile.close()
                if (iparttypes == 1):
                    tfile = h5py.File(tfilename, 'r')
                    utfile = h5py.File(utfilename, 'r')
                    tdata = np.uint16(tfile["Particle_types"])
                    utdata = np.uint16(utfile["Particle_types"])
                    tfile.close()
                    utfile.close()

            # now with data loaded, process it to produce data structure
            particledata['Npart'][counter:counter+numhalos] = numingroup
            unumingroup = np.zeros(numhalos, dtype=np.uint64)
            for i in range(int(numhalos-1)):
                unumingroup[i] = (uoffset[i+1]-uoffset[i])
            unumingroup[-1] = (unpart-uoffset[-1])
            particledata['Npart_unbound'][counter:counter +
                                          numhalos] = unumingroup
            for i in range(numhalos):
                particledata['Particle_IDs'][int(
                    i+counter)] = np.zeros(numingroup[i], dtype=np.int64)
                particledata['Particle_IDs'][int(i+counter)][:int(
                    numingroup[i]-unumingroup[i])] = piddata[offset[i]:offset[i]+numingroup[i]-unumingroup[i]]
                particledata['Particle_IDs'][int(
                    i+counter)][int(numingroup[i]-unumingroup[i]):numingroup[i]] = upiddata[uoffset[i]:uoffset[i]+unumingroup[i]]
                if (iparttypes == 1):
                    particledata['Particle_Types'][int(
                        i+counter)] = np.zeros(numingroup[i], dtype=np.int64)
                    particledata['Particle_Types'][int(i+counter)][:int(
                        numingroup[i]-unumingroup[i])] = tdata[offset[i]:offset[i]+numingroup[i]-unumingroup[i]]
                    particledata['Particle_Types'][int(
                        i+counter)][int(numingroup[i]-unumingroup[i]):numingroup[i]] = utdata[uoffset[i]:uoffset[i]+unumingroup[i]]
            counter += numhalos

    return particledata


def ReadSOParticleDataFile(basefilename, ibinary=0, iparttypes=0, iverbose=0, binarydtype=np.int64):
    """
    VELOCIraptor/STF catalog_group, catalog_particles and catalog_parttypes in various formats

    Note that a file will indicate how many files the total output has been split into

    """
    inompi = True
    if (iverbose):
        print("reading particle data", basefilename)
    filename = basefilename+".catalog_SOlist"
    # check for file existence
    if (os.path.isfile(filename) == True):
        numfiles = 0
    else:
        filename += ".0"
        inompi = False
        if (os.path.isfile(filename) == False):
            print("file not found", filename)
            return []
    byteoffset = 0

    # load header information from file to get total number of groups
    # ascii
    if (ibinary == 0):
        gfile = open(filename, 'r')
        [filenum, numfiles] = gfile.readline().split()
        filenum = int(filenum)
        numfiles = int(numfiles)
        [numSO, numtotSO] = gfile.readline().split()
        [numparts, numtotparts] = gfile.readline().split()
        numSO = np.uint64(numSO)
        numtothalos = np.uint64(numtotSO)
        numparts = np.uint64(numparts)
        numtotparts = np.uint64(numtotparts)
    # binary
    elif (ibinary == 1):
        gfile = open(filename, 'rb')
        [filenum, numfiles] = np.fromfile(gfile, dtype=np.int32, count=2)
        [numSO, numtotSO] = np.fromfile(gfile, dtype=np.uint64, count=2)
        [numparts, numtotparts] = np.fromfile(gfile, dtype=np.uint64, count=2)
    # hdf
    elif (ibinary == 2):
        gfile = h5py.File(filename, 'r')
        filenum = int(gfile["File_id"][0])
        numfiles = int(gfile["Num_of_files"][0])
        numSO = np.uint64(gfile["Num_of_SO_regions"][0])
        numtotSO = np.uint64(gfile["Total_num_of_SO_regions"][0])
        numparts = np.uint64(gfile["Num_of_particles_in_SO_regions"][0])
        numtotparts = np.uint64(
            gfile["Total_num_of_particles_in_SO_regions"][0])
    gfile.close()
    particledata = dict()
    particledata['Npart'] = []
    particledata['Particle_IDs'] = []
    if (iparttypes):
        particledata['Particle_Types'] = []
    if (iverbose):
        print("SO lists contains ", numtotSO, " regions containing total of ",
              numtotparts, " in ", numfiles, " files")
    if (numtotSO == 0):
        return particledata
    particledata['Npart'] = np.zeros(numtotSO, dtype=np.uint64)
    particledata['Particle_IDs'] = [[] for i in range(numtotSO)]
    if (iparttypes):
        particledata['Particle_Types'] = [[] for i in range(numtotSO)]

    # now for all files
    counter = np.uint64(0)
    for ifile in range(numfiles):
        filename = basefilename+".catalog_SOlist"
        if (inompi == False):
            filename += "."+str(ifile)
        # ascii
        if (ibinary == 0):
            gfile = open(filename, 'r')
            # read header information
            gfile.readline()
            [numSO, foo] = gfile.readline().split()
            [numparts, foo] = gfile.readline().split()
            numSO = np.uint64(numSO)
            numparts = np.uint64(numSO)
            gfile.close()
            # load data
            gdata = np.loadtxt(filename, skiprows=2, dtype=np.uint64)
            numingroup = gdata[:numSO]
            offset = gdata[np.int64(numSO):np.int64(2*numSO)]
            piddata = gdata[np.int64(2*numSO):np.int64(2*numSO+numparts)]
            if (iparttypes):
                tdata = gdata[np.int64(2*numSO+numparts):np.int64(2*numSO+2*numparts)]
        # binary
        elif (ibinary == 1):
            gfile = open(filename, 'rb')
            np.fromfile(gfile, dtype=np.int32, count=2)
            [numSO, foo] = np.fromfile(gfile, dtype=np.uint64, count=2)
            [numparts, foo] = np.fromfile(gfile, dtype=np.uint64, count=2)
            numingroup = np.fromfile(gfile, dtype=binarydtype, count=numSO)
            offset = np.fromfile(gfile, dtype=binarydtype, count=numSO)
            piddata = np.fromfile(gfile, dtype=binarydtype, count=numparts)
            if (iparttypes):
                tdata = np.fromfile(gfile, dtype=np.int32, count=numparts)
            gfile.close()
        # hdf
        elif (ibinary == 2):
            gfile = h5py.File(filename, 'r')
            numSO = np.uint64(gfile["Num_of_SO_regions"][0])
            numingroup = np.uint64(gfile["SO_size"])
            offset = np.uint64(gfile["Offset"])
            piddata = np.int64(gfile["Particle_IDs"])
            if (iparttypes):
                tdata = np.int64(gfile["Particle_types"])

            gfile.close()

        # now with data loaded, process it to produce data structure
        particledata['Npart'][counter:counter+numSO] = numingroup
        for i in range(numSO):
            particledata['Particle_IDs'][int(
                i+counter)] = np.array(piddata[offset[i]:offset[i]+numingroup[i]])
        if (iparttypes):
            for i in range(numSO):
                particledata['Particle_types'][int(
                    i+counter)] = np.array(tdata[offset[i]:offset[i]+numingroup[i]])
        counter += numSO

    return particledata

def ReadProfilesFile(basefilename, ibinary=2, iseparatesubfiles=0, iverbose=0):
    inompi = True
    if (iverbose):
        print("reading radial profile data", basefilename)
    filename = basefilename+".profiles"
    # check for file existence
    if (os.path.isfile(filename) == True):
        numfiles = 0
    else:
        filename += ".0"
        inompi = False
        if (os.path.isfile(filename) == False):
            print("file not found", filename)
            return []
    byteoffset = 0
    if (ibinary !=2):
        print('WARNING: ASCII and raw binary interface to reading profiles not yet implmented')
        print('Returning empty dictionary')
        return None
    profiledata = dict()

    # hdf
    if (ibinary == 2):
        gfile = h5py.File(filename, 'r')
        filenum = int(gfile["File_id"][0])
        numfiles = int(gfile["Num_of_files"][0])
        numhalos = np.uint64(gfile["Num_of_halos"][0])
        numgroups = np.uint64(gfile["Num_of_groups"][0])
        numtothalos = np.uint64(gfile["Total_num_of_halos"][0])
        numtotgroups = np.uint64(gfile["Total_num_of_groups"][0])
        profiledata['Total_num_of_halos'] = numtothalos
        profiledata['Total_num_of_groups'] = numtotgroups
        profiledata['Radial_bin_edges'] = np.array(gfile["Radial_bin_edges"])
        allkeys = list(gfile.keys())
        gfile.close()
    if (numtothalos == 0):
        return profiledata

    # get list of active keys
    props = ['Mass', 'Npart']
    proptypes = ['_profile', '_inclusive_profile']
    parttypes = ['', '_gas', '_gas_sf', '_gas_nsf', '_star', '_dm', '_interloper']
    loadablekeys = []
    for prop in props:
        for proptype in proptypes:
            for parttype in parttypes:
                key = prop+proptype+parttype
                if (key in allkeys):
                    loadablekeys.append(key)

    # now for all files
    counter = np.uint64(0)
    for ifile in range(numfiles):
        filename = basefilename+".profiles"
        if (inompi == False):
            filename += "."+str(ifile)
        if (ibinary == 2):
            gfile = h5py.File(filename, 'r')
            numhalos = np.uint64(gfile["Num_of_halos"][0])
            numgroups = np.uint64(gfile["Num_of_groups"][0])
            for key in loadablekeys:
                profiledata[key] = np.array(gfile[key])
            gfile.close()

    return profiledata

"""
    Routines to build a hierarchy structure (both spatially and temporally)
"""


def BuildHierarchy(halodata, iverbose=0):
    """
    the halo data stored in a velociraptor .properties file should store the id of its parent halo. Here
    this catalog is used to produce a hierarchy to quickly access the relevant subhaloes of a parent halo.
    #todo this should be deprecated as Hierarchy information is typically already contained in halo information
    """
    halohierarchy = []
    start = time.process_time()
    if (iverbose):
        print("setting hierarchy")
    numhalos = len(halodata["npart"])
    subhaloindex = np.where(halodata["hostHaloID"] != -1)
    lensub = len(subhaloindex[0])
    haloindex = np.where(halodata["hostHaloID"] == -1)
    lenhal = len(haloindex[0])
    halohierarchy = [[] for k in range(numhalos)]
    if (iverbose):
        print("prelims done ", time.process_time()-start)
    for k in range(lenhal):
        halohierarchy[haloindex[0][k]] = np.where(
            halodata["hostHaloID"] == halodata["ID"][haloindex[0][k]])
    # NOTE: IMPORTANT this is only adding the subsub halos! I need to eventually parse the hierarchy
    # data first to deteremine the depth of the subhalo hierarchy and store how deep an object is in the hierarchy
    # then I can begin adding (sub)subhalos to parent subhalos from the bottom level up
    """
    for k in range(0, len(halodata["npart"])):
        hid = np.int32(halodata["hostHaloID"][k])
        if (hid>-1 and halohierarchy[k]!= []):
            halohierarchy[hid] = np.append(np.int32(halohierarchy[hid]), halohierarchy[k])
    """
    if (iverbose):
        print("hierarchy set in read in ", time.process_time()-start)
    return halohierarchy


def TraceMainProgen(istart, ihalo, numsnaps, numhalos, halodata, tree, TEMPORALHALOIDVAL):
    """
    Follows a halo along tree to identify main progenitor
    """
    # start at this snapshot
    k = istart
    # see if halo does not have a tail (descendant set).
    if (halodata[k]['Tail'][ihalo] == 0):
        # if halo has not had a tail set the branch needs to be walked along the main branch
        haloid = halodata[k]['ID'][ihalo]
        # only set the head if it has not been set
        # otherwise it should have already been set and just need to store the root head
        if (halodata[k]['Head'][ihalo] == 0):
            halodata[k]['Head'][ihalo] = haloid
            halodata[k]['HeadSnap'][ihalo] = k
            halodata[k]['RootHead'][ihalo] = haloid
            halodata[k]['RootHeadSnap'][ihalo] = k
            roothead, rootsnap, rootindex = haloid, k, ihalo
        else:
            roothead = halodata[k]['RootHead'][ihalo]
            rootsnap = halodata[k]['RootHeadSnap'][ihalo]
            rootindex = int(roothead % TEMPORALHALOIDVAL)-1
        # now move along tree first pass to store head and tails and root heads of main branch
        while (True):
            # instead of seraching array make use of the value of the id as it should be in id order
            #wdata = np.where(tree[k]['haloID'] ==  haloid)
            #w2data = np.where(halodata[k]['ID'] ==  haloid)[0][0]
            wdata = w2data = int(haloid % TEMPORALHALOIDVAL)-1
            halodata[k]['Num_progen'][wdata] = tree[k]['Num_progen'][wdata]
            # if no more progenitors, break from search
            # if (tree[k]['Num_progen'][wdata[0][0]] ==  0 or len(wdata[0]) ==  0):
            if (tree[k]['Num_progen'][wdata] == 0):
                # store for current halo its tail and root tail info (also store root tail for root head)
                halodata[k]['Tail'][w2data] = haloid
                halodata[k]['TailSnap'][w2data] = k
                halodata[k]['RootTail'][w2data] = haloid
                halodata[k]['RootTailSnap'][w2data] = k
                # only set the roots tail if it has not been set before (ie: along the main branch of root halo)
                # if it has been set then we are walking along a secondary branch of the root halo's tree
                if (halodata[rootsnap]['RootTail'][rootindex] == 0):
                    halodata[rootsnap]['RootTail'][rootindex] = haloid
                    halodata[rootsnap]['RootTailSnap'][rootindex] = k
                break

            # store main progenitor
            #mainprog = tree[k]['Progen'][wdata[0][0]][0]
            mainprog = tree[k]['Progen'][wdata][0]
            # calculate stepsize based on the halo ids
            stepsize = int(((haloid-haloid % TEMPORALHALOIDVAL) -
                            (mainprog-mainprog % TEMPORALHALOIDVAL))/TEMPORALHALOIDVAL)
            # store tail
            halodata[k]['Tail'][w2data] = mainprog
            halodata[k]['TailSnap'][w2data] = k+stepsize
            k += stepsize

            # instead of searching array make use of the value of the id as it should be in id order
            # for progid in tree[k-stepsize]['Progen'][wdata[0][0]]:
            #wdata3 = np.where(halodata[k]['ID'] ==  progid)[0][0]
            for progid in tree[k-stepsize]['Progen'][wdata]:
                wdata3 = int(progid % TEMPORALHALOIDVAL)-1
                halodata[k]['Head'][wdata3] = haloid
                halodata[k]['HeadSnap'][wdata3] = k-stepsize
                halodata[k]['RootHead'][wdata3] = roothead
                halodata[k]['RootHeadSnap'][wdata3] = rootsnap

            # then store next progenitor
            haloid = mainprog


def TraceMainProgenParallelChunk(istart, ihalochunk, numsnaps, numhalos, halodata, tree, TEMPORALHALOIDVAL):
    """
    Wrapper to allow for parallelisation
    """
    for ihalo in ihalochunk:
        TraceMainProgen(istart, ihalo, numsnaps, numhalos,
                        halodata, tree, TEMPORALHALOIDVAL)


def BuildTemporalHeadTail(numsnaps, tree, numhalos, halodata, TEMPORALHALOIDVAL=1000000000000, iverbose=1):
    """
    Adds for each halo its Head and Tail and stores Roothead and RootTail to the halo
    properties file
    TEMPORALHALOIDVAL is used to parse the halo ids and determine the step size between descendant and progenitor
    """
    print("Building Temporal catalog with head and tails")
    sys.stdout.flush()

    if("HaloID_snapshot_offset" in tree["Header"]):
        snapshotoffset = tree["Header"]["HaloID_snapshot_offset"]
    else:
        snapshotoffset = 0

    for k in range(snapshotoffset,snapshotoffset+numsnaps):
        halodata[k]['Head'] = np.zeros(numhalos[k], dtype=np.int64)
        halodata[k]['Tail'] = np.zeros(numhalos[k], dtype=np.int64)
        halodata[k]['HeadSnap'] = np.zeros(numhalos[k], dtype=np.int32)
        halodata[k]['TailSnap'] = np.zeros(numhalos[k], dtype=np.int32)
        halodata[k]['RootHead'] = np.zeros(numhalos[k], dtype=np.int64)
        halodata[k]['RootTail'] = np.zeros(numhalos[k], dtype=np.int64)
        halodata[k]['RootHeadSnap'] = np.zeros(numhalos[k], dtype=np.int32)
        halodata[k]['RootTailSnap'] = np.zeros(numhalos[k], dtype=np.int32)
        halodata[k]['Num_progen'] = np.zeros(numhalos[k], dtype=np.uint32)
        # aliases
        halodata[k]['Progenitor'] = halodata[k]['Tail']
        halodata[k]['Descendant'] = halodata[k]['Head']
        halodata[k]['DescendantSnap'] = halodata[k]['HeadSnap']
        halodata[k]['ProgenitorSnap'] = halodata[k]['TailSnap']
        halodata[k]['RootProgenitor'] = halodata[k]['RootTail']
        halodata[k]['RootProgenitorSnap'] = halodata[k]['RootTailSnap']
        halodata[k]['RootDescendant'] = halodata[k]['RootHead']
        halodata[k]['RootDescendantSnap'] = halodata[k]['RootHeadSnap']
    # for each snapshot identify halos that have not had their tail set
    # for these halos, the main branch must be walked
    # allocate python manager to wrapper the tree and halo catalog so they can be altered in parallel
    manager = mp.Manager()
    chunksize = 5000000  # have each thread handle this many halos at once
    # init to that at this point snapshots should be run in parallel
    if (numhalos[0] > 2*chunksize):
        iparallel = 1
    else:
        iparallel = -1  # no parallel at all
    iparallel = -1

    totstart = time.process_time()

    if (iparallel == 1):
        # need to copy halodata as this will be altered
        if (iverbose > 0):
            print("copying halo")
            sys.stdout.flush()
        start = time.process_time()
        mphalodata = manager.list([manager.dict(halodata[k])
                                   for k in range(snapshotoffset,snapshotoffset+numsnaps)])
        if (iverbose > 0):
            print("done", time.process_time()-start)
            sys.stdout.flush()

    for istart in range(snapshotoffset,snapshotoffset+numsnaps):
        if (iverbose > 0):
            print("Starting from halos at ", istart, "with", numhalos[istart])
            sys.stdout.flush()
        if (numhalos[istart] == 0):
            continue
        # if the number of halos is large then run in parallel
        if (numhalos[istart] > 2*chunksize and iparallel == 1):
            # determine maximum number of threads
            nthreads = int(min(mp.cpu_count(), np.ceil(
                numhalos[istart]/float(chunksize))))
            nchunks = int(
                np.ceil(numhalos[istart]/float(chunksize)/float(nthreads)))
            if (iverbose > 0):
                print("Using", nthreads, "threads to parse ",
                      numhalos[istart], " halos in ", nchunks, "chunks, each of size", chunksize)
                sys.stdout.flush()
            # now for each chunk run a set of proceses
            for j in range(nchunks):
                start = time.process_time()
                offset = j*nthreads*chunksize
                # if last chunk then must adjust nthreads
                if (j == nchunks-1):
                    nthreads = int(
                        np.ceil((numhalos[istart]-offset)/float(chunksize)))

                halochunk = [range(offset+k*chunksize, offset+(k+1)*chunksize)
                             for k in range(nthreads)]
                # adjust last chunk
                if (j == nchunks-1):
                    halochunk[-1] = range(offset+(nthreads-1)
                                          * chunksize, numhalos[istart])
                # when calling a process pass not just a work queue but the pointers to where data should be stored
                processes = [mp.Process(target=TraceMainProgenParallelChunk, args=(
                    istart, halochunk[k], numsnaps, numhalos, mphalodata, tree, TEMPORALHALOIDVAL)) for k in range(nthreads)]
                count = 0
                for p in processes:
                    print(count+offset, k,
                          min(halochunk[count]), max(halochunk[count]))
                    sys.stdout.flush()
                    p.start()
                    count += 1
                for p in processes:
                    # join thread and see if still active
                    p.join()
                if (iverbose > 1):
                    print((offset+j*nthreads*chunksize) /
                          float(numhalos[istart]), " done in", time.process_time()-start)
                    sys.stdout.flush()
        # otherwise just single
        else:
            # if first time entering non parallel section copy data back from parallel manager based structure to original data structure
            # as parallel structures have been updated
            if (iparallel == 1):
                #tree = [dict(mptree[k]) for k in range(numsnaps)]
                halodata = [dict(mphalodata[k]) for k in range(snapshotoffset,snapshotoffset+numsnaps)]
                # set the iparallel flag to 0 so that all subsequent snapshots (which should have fewer objects) not run in parallel
                # this is principly to minimize the amount of copying between manager based parallel structures and the halo/tree catalogs
                iparallel = 0
            start = time.process_time()
            chunksize = max(int(0.10*numhalos[istart]), 10)
            for j in range(numhalos[istart]):
                # start at this snapshot
                #start = time.process_time()
                TraceMainProgen(istart, j, numsnaps, numhalos,
                                halodata, tree, TEMPORALHALOIDVAL)
                if (j % chunksize == 0 and j > 0):
                    if (iverbose > 1):
                        print(
                            "done", j/float(numhalos[istart]), "in", time.process_time()-start)
                        sys.stdout.flush()
                    start = time.process_time()
    if (iverbose > 0):
        print("done with first bit")
        sys.stdout.flush()
    # now have walked all the main branches and set the root head, head and tail values
    # and can set the root tail of all halos. Start at end of the tree and move in reverse setting the root tail
    # of a halo's head so long as that halo's tail is the current halo (main branch)
    for istart in range(snapshotoffset+numsnaps-1, snapshotoffset-1, -1):
        for j in range(numhalos[istart]):
            # if a halo's root tail is itself then start moving up its along to its head (if its head is not itself as well
            k = istart
            #rootheadid, rootheadsnap = halodata[k]['RootHead'][j], halodata[k]['RootHeadSnap'][j]
            roottailid, roottailsnap = halodata[k]['RootTail'][j], halodata[k]['RootTailSnap'][j]
            headid, headsnap = halodata[k]['Head'][j], halodata[k]['HeadSnap'][j]
            if (roottailid == halodata[k]['ID'][j] and headid != halodata[k]['ID'][j]):
                #headindex = np.where(halodata[headsnap]['ID'] ==  headid)[0][0]
                headindex = int(headid % TEMPORALHALOIDVAL)-1
                headtailid, headtailsnap = halodata[headsnap]['Tail'][
                    headindex], halodata[headsnap]['TailSnap'][headindex]
                haloid = halodata[k]['ID'][j]
                # only proceed in setting root tails of a head who's tail is the same as halo (main branch) till we reach a halo who is its own head
                while (headtailid == haloid and headid != haloid):
                    # set root tails
                    halodata[headsnap]['RootTail'][headindex] = roottailid
                    halodata[headsnap]['RootTailSnap'][headindex] = roottailsnap
                    # move to next head
                    haloid = halodata[headsnap]['ID'][headindex]
                    #haloindex = np.where(halodata[headsnap]['ID'] ==  haloid)[0][0]
                    haloindex = int(haloid % TEMPORALHALOIDVAL)-1
                    halosnap = headsnap
                    headid, headsnap = halodata[halosnap]['Head'][haloindex], halodata[halosnap]['HeadSnap'][haloindex]
                    headindex = int(headid % TEMPORALHALOIDVAL)-1
                    # store the tail of the next head
                    headtailid, headtailsnap = halodata[headsnap]['Tail'][
                        headindex], halodata[headsnap]['TailSnap'][headindex]
    print("Done building", time.process_time()-totstart)
    sys.stdout.flush()


def TraceMainDescendant(istart, ihalo, numsnaps, numhalos, halodata, tree, TEMPORALHALOIDVAL, ireverseorder=False):
    """
    Follows a halo along descendant tree to root tails
    if reverse order than late times start at 0 and as one moves up in index
    one moves backwards in time
    """

    # start at this snapshot
    halosnap = istart

    # see if halo does not have a Head set
    if (halodata[halosnap]['Head'][ihalo] == 0):
        # if halo has not had a Head set the branch needs to be walked along the main branch
        haloid = halodata[halosnap]['ID'][ihalo]
        # only set the Root Tail if it has not been set. Here if halo has not had
        # tail set, then must be the the first progenitor
        # otherwise it should have already been set and just need to store the root tail
        if (halodata[halosnap]['Tail'][ihalo] == 0):
            halodata[halosnap]['Tail'][ihalo] = haloid
            halodata[halosnap]['TailSnap'][ihalo] = halosnap
            halodata[halosnap]['RootTail'][ihalo] = haloid
            halodata[halosnap]['RootTailSnap'][ihalo] = halosnap
            roottail, rootsnap, rootindex = haloid, halosnap, ihalo
        else:
            roottail = halodata[halosnap]['RootTail'][ihalo]
            rootsnap = halodata[halosnap]['RootTailSnap'][ihalo]
            rootindex = int(roottail % TEMPORALHALOIDVAL)-1
        # now move along tree first pass to store head and tails and root tails of main branch
        while (True):
            # ids contain index information
            haloindex = int(haloid % TEMPORALHALOIDVAL)-1

            halodata[halosnap]['Num_descen'][haloindex] = tree[halosnap]['Num_descen'][haloindex]
            # if no more descendants, break from search
            if (halodata[halosnap]['Num_descen'][haloindex] == 0):
                # store for current halo its tail and root tail info (also store root tail for root head)
                halodata[halosnap]['Head'][haloindex] = haloid
                halodata[halosnap]['HeadSnap'][haloindex] = halosnap
                halodata[halosnap]['RootHead'][haloindex] = haloid
                halodata[halosnap]['RootHeadSnap'][haloindex] = halosnap
                rootheadid, rootheadsnap, rootheadindex = haloid, halosnap, haloindex
                # only set the roots head of the root tail
                # if it has not been set before (ie: along the main branch of root halo)
                if (halodata[rootsnap]['RootHead'][rootindex] == 0):
                    halosnap, haloindex, haloid = rootsnap, rootindex, roottail
                    # set the root head of the main branch
                    while(True):
                        halodata[halosnap]['RootHead'][haloindex] = rootheadid
                        halodata[halosnap]['RootHeadSnap'][haloindex] = rootheadsnap
                        descen = halodata[halosnap]['Head'][haloindex]
                        descenindex = int(descen % TEMPORALHALOIDVAL)-1
                        descensnap = int(
                            ((descen-descen % TEMPORALHALOIDVAL))/TEMPORALHALOIDVAL)
                        if (ireverseorder):
                            descensnap = numsnaps-1-descensnap
                        if (haloid == descen):
                            break
                        halosnap, haloindex, haloid = descensnap, descenindex, descen
                break
            # now store the rank of the of the descandant.
            descenrank = tree[halosnap]['Rank'][haloindex][0]
            halodata[halosnap]['HeadRank'][haloindex] = descenrank
            # as we are only moving along main branches stop if object is rank is not 0
            if (descenrank > 0):
                break
            # otherwise, get the descendant
            # store main progenitor
            maindescen = tree[halosnap]['Descen'][haloindex][0]
            maindescenindex = int(maindescen % TEMPORALHALOIDVAL)-1
            maindescensnap = int(
                ((maindescen-maindescen % TEMPORALHALOIDVAL))/TEMPORALHALOIDVAL)
            # if reverse order, then higher snap values correspond to lower index
            if (ireverseorder):
                maindescensnap = numsnaps-1-maindescensnap
            # calculate stepsize in time based on the halo ids
            stepsize = maindescensnap-halosnap

            # store descendant
            halodata[halosnap]['Head'][haloindex] = maindescen
            halodata[halosnap]['HeadSnap'][haloindex] = maindescensnap

            # and update the root tails of the object
            halodata[maindescensnap]['Tail'][maindescenindex] = haloid
            halodata[maindescensnap]['TailSnap'][maindescenindex] = halosnap
            halodata[maindescensnap]['RootTail'][maindescenindex] = roottail
            halodata[maindescensnap]['RootTailSnap'][maindescenindex] = rootsnap
            halodata[maindescensnap]['Num_progen'][maindescenindex] += 1

            # then move to the next descendant
            haloid = maindescen
            halosnap = maindescensnap


def TraceMainDescendantParallelChunk(istart, ihalochunk, numsnaps, numhalos, halodata, tree, TEMPORALHALOIDVAL, ireverseorder):
    for ihalo in ihalochunk:
        TraceMainDescendant(istart, ihalo, numsnaps, numhalos,
                            halodata, tree, TEMPORALHALOIDVAL, ireverseorder)


def BuildTemporalHeadTailDescendant(numsnaps, tree, numhalos, halodata, TEMPORALHALOIDVAL=1000000000000, ireverseorder=False, iverbose=1):
    """
    Adds for each halo its Head and Tail and stores Roothead and RootTail to the halo
    properties file
    TEMPORALHALOIDVAL is used to parse the halo ids and determine the step size between descendant and progenitor
    """
    print("Building Temporal catalog with head and tails using a descendant tree")
    #store if merit present in the raw tree
    imerit=('Merit' in tree[0].keys())

    if("HaloID_snapshot_offset" in tree["Header"]):
        snapshotoffset = tree["Header"]["HaloID_snapshot_offset"]
    else:
        snapshotoffset = 0

    for k in range(snapshotoffset,snapshotoffset+numsnaps):

        #Set the VELOCIraptor ID to point to the TreeFrog ID
        halodata[k]["ID"] = tree[k]["haloID"]

        #Intialize the rest of the dataset
        halodata[k]['Head'] = np.zeros(numhalos[k], dtype=np.int64)
        halodata[k]['Tail'] = np.zeros(numhalos[k], dtype=np.int64)
        halodata[k]['HeadSnap'] = np.zeros(numhalos[k], dtype=np.int32)
        halodata[k]['TailSnap'] = np.zeros(numhalos[k], dtype=np.int32)
        halodata[k]['HeadIndex'] = np.zeros(numhalos[k], dtype=np.int64)
        halodata[k]['TailIndex'] = np.zeros(numhalos[k], dtype=np.int64)
        halodata[k]['RootHead'] = np.zeros(numhalos[k], dtype=np.int64)
        halodata[k]['RootTail'] = np.zeros(numhalos[k], dtype=np.int64)
        halodata[k]['RootHeadSnap'] = np.zeros(numhalos[k], dtype=np.int32)
        halodata[k]['RootTailSnap'] = np.zeros(numhalos[k], dtype=np.int32)
        halodata[k]['RootHeadIndex'] = np.zeros(numhalos[k], dtype=np.int64)
        halodata[k]['RootTailIndex'] = np.zeros(numhalos[k], dtype=np.int64)
        halodata[k]['HeadRank'] = np.zeros(numhalos[k], dtype=np.int64)
        halodata[k]['Num_descen'] = np.zeros(numhalos[k], dtype=np.uint32)
        halodata[k]['Num_progen'] = np.zeros(numhalos[k], dtype=np.uint32)
    # for each snapshot identify halos that have not had their tail set
    # for these halos, the main branch must be walked
    # allocate python manager to wrapper the tree and halo catalog so they can be altered in parallel
    manager = mp.Manager()
    chunksize = 5000000  # have each thread handle this many halos at once
    # init to that at this point snapshots should be run in parallel
    if (numhalos[0] > 2*chunksize):
        iparallel = 1
    else:
        iparallel = -1  # no parallel at all
    iparallel = -1

    totstart = time.process_time()
    start0=time.process_time()

    if (ireverseorder):
        snaplist = range(snapshotoffset+numsnaps-1, snapshotoffset-1, -1)
    else:
        snaplist = range(snapshotoffset,snapshotoffset+numsnaps)
    for istart in snaplist:
        start2=time.process_time()
        if (iverbose > 0):
            print('starting head/tail at snapshot ', istart, ' containing ', numhalos[istart], 'halos')
        if (numhalos[istart] == 0): continue
        #set tails and root tails if necessary
        wdata = np.where(halodata[istart]['Tail'] == 0)[0]
        numareroottails = wdata.size
        if (iverbose > 0):
            print(numareroottails,' halos are root tails ')
            sys.stdout.flush()
        if (numareroottails > 0):
            halodata[istart]['Tail'][wdata] = np.array(halodata[istart]['ID'][wdata],copy=True)
            halodata[istart]['RootTail'][wdata] = np.array(halodata[istart]['ID'][wdata],copy=True)
            halodata[istart]['TailSnap'][wdata] = istart*np.ones(wdata.size, dtype=np.int32)
            halodata[istart]['RootTailSnap'][wdata] = istart*np.ones(wdata.size, dtype=np.int32)
            halodata[istart]['TailIndex'][wdata] = np.array(halodata[istart]['ID'][wdata]% TEMPORALHALOIDVAL - 1,dtype=np.int64,copy=True)
            halodata[istart]['RootTailIndex'][wdata] = np.array(halodata[istart]['ID'][wdata]% TEMPORALHALOIDVAL - 1,dtype=np.int64,copy=True)
        #init heads to ids
        halodata[istart]['Head'] = np.array(halodata[istart]['ID'],copy=True)
        halodata[istart]['HeadSnap'] = istart*np.ones(numhalos[istart],dtype=np.int32)
        halodata[istart]['HeadIndex'] = np.array(halodata[istart]['ID']% TEMPORALHALOIDVAL - 1, dtype=np.int64,copy=True)
        #find all halos that have descendants and set there heads
        if (istart == numsnaps-1):
            halodata[istart]['RootHead'] = np.array(halodata[istart]['ID'],copy=True)
            halodata[istart]['RootHeadSnap'] = istart*np.ones(numhalos[istart], dtype=np.int32)
            halodata[istart]['RootHeadIndex'] = np.array(halodata[istart]['ID']% TEMPORALHALOIDVAL - 1, dtype=np.int64,copy=True)
            continue
        wdata = None
        descencheck=(tree[istart]['Num_descen']>0)
        wdata=np.where(descencheck)[0]
        numwithdescen = wdata.size
        if (iverbose > 0):
            print(numwithdescen, 'have descendants')
            sys.stdout.flush()
        if (numwithdescen>0):
            # should figure out how to best speed this up
            ranks = np.array([tree[istart]['Rank'][index][0] for index in wdata], dtype=np.int32)
            descenids = np.array([tree[istart]['Descen'][index][0] for index in wdata], dtype=np.int64)
            descenindex = np.array(descenids % TEMPORALHALOIDVAL - 1, dtype=np.int64)
            if (imerit):
                activemerits = np.array([tree[istart]['Merit'][index][0] for index in wdata], dtype=np.int64)
            # rest is quick
            descensnaps = np.array((descenids - descenindex - np.int64(1)) / TEMPORALHALOIDVAL, dtype=np.int32)
            if (ireverseorder):
                descensnaps = numsnaps - 1 - descensnaps
            halodata[istart]['HeadRank'][wdata] = np.array(ranks, copy=True)
            halodata[istart]['Head'][wdata] = np.array(descenids, copy=True)
            halodata[istart]['HeadSnap'][wdata] = np.array(descensnaps, copy=True)
            halodata[istart]['HeadIndex'][wdata] = np.array(descenindex, copy=True)
            # showld figure out how to speed this up
            for i in range(numwithdescen):
                isnap, idescenindex = descensnaps[i], descenindex[i]
                halodata[isnap]['Num_progen'][idescenindex] += 1
            # set the tails of all these objects and their root tails as well
            wdata2 = np.where(ranks == 0)[0]
            numactive = wdata2.size
            if (numactive>0):
                activetails = wdata[wdata2]
                descensnaps = descensnaps[wdata2]
                descenindex = descenindex[wdata2]
                if (imerit):
                    activemerits = activemerits[wdata2]
                # should parallelise this
                for i in range(numactive):
                    index, isnap, idescenindex = activetails[i], descensnaps[i], descenindex[i]
                    #add check to see if this root head has already been assigned, then may have an error in the
                    #the mpi mesh point
                    if (halodata[isnap]['Tail'][idescenindex] == 0):
                        halodata[isnap]['Tail'][idescenindex] = halodata[istart]['ID'][index]
                        halodata[isnap]['RootTail'][idescenindex] = halodata[istart]['RootTail'][index]
                        halodata[isnap]['TailSnap'][idescenindex] = istart
                        halodata[isnap]['RootTailSnap'][idescenindex] = halodata[istart]['RootTailSnap'][index]
                        halodata[isnap]['TailIndex'][idescenindex] = index
                        halodata[isnap]['RootTailIndex'][idescenindex] = halodata[istart]['RootTailIndex'][index]
                    #if tail was assigned then need to compare merits and designed which one to use
                    else:
                        #if can compare merits
                        if (imerit):
                            curMerit = activemerits[i]
                            prevTailIndex = halodata[isnap]['TailIndex'][idescenindex]
                            prevTailSnap = halodata[isnap]['TailSnap'][idescenindex]
                            compMerit = tree[prevTailSnap]['Merit'][prevTailIndex][0]
                            if (curMerit > compMerit):
                                halodata[prevTailSnap]['HeadRank'][prevTailIndex]+=1
                                halodata[isnap]['Tail'][idescenindex] = halodata[istart]['ID'][index]
                                halodata[isnap]['RootTail'][idescenindex] = halodata[istart]['RootTail'][index]
                                halodata[isnap]['TailSnap'][idescenindex] = istart
                                halodata[isnap]['RootTailSnap'][idescenindex] = halodata[istart]['RootTailSnap'][index]
                                halodata[isnap]['TailIndex'][idescenindex] = index
                                halodata[isnap]['RootTailIndex'][idescenindex] = halodata[istart]['RootTailIndex'][index]
                            else:
                                halodata[istart]['HeadRank'][index]=1
                        #if merits not present then assume first connection found is better
                        else:
                            halodata[istart]['HeadRank'][index]=1
            wdata2 = None
            descenids = None
            descensnaps = None
            descenindex = None
        #set root heads of things that have no descendants
        wdata = np.where(descencheck == False)[0]
        if (wdata.size > 0):
            halodata[istart]['RootHead'][wdata] = np.array(halodata[istart]['ID'][wdata], copy=True)
            halodata[istart]['RootHeadSnap'][wdata] = istart*np.ones(wdata.size, dtype=np.int32)
            halodata[istart]['RootHeadIndex'][wdata] = np.array(halodata[istart]['ID'][wdata]% TEMPORALHALOIDVAL - 1, dtype=np.int64, copy=True)
        wdata = None
        descencheck = None
        if (iverbose > 0):
            print('finished in', time.process_time()-start2)
            sys.stdout.flush()
    if (iverbose > 0):
        print("done with first bit, setting the main branches walking backward",time.process_time()-start0)
        sys.stdout.flush()
    # now have walked all the main branches and set the root tail, head and tail values
    # in case halo data is with late times at beginning need to process items in reverse
    if (ireverseorder):
        snaplist = range(snapshotoffset,snapshotoffset+numsnaps)
    else:
        snaplist = range(snapshotoffset+numsnaps-1, snapshotoffset-1, -1)
    # first set root heads of main branches
    for istart in snaplist:
        if (numhalos[istart] == 0):
            continue
        wdata = np.where((halodata[istart]['RootHead'] != 0))[0]
        numactive=wdata.size
        if (iverbose > 0):
            print('Setting root heads at ', istart, 'halos', numhalos[istart], 'active', numactive)
            sys.stdout.flush()
        if (numactive == 0):
            continue

        haloidarray = halodata[istart]['Tail'][wdata]
        haloindexarray = halodata[istart]['TailIndex'][wdata]
        halosnaparray = halodata[istart]['TailSnap'][wdata]

        if (ireverseorder):
            halosnaparray = numsnaps - 1 - halosnaparray
        # go to root tails and walk the main branch
        for i in np.arange(numactive,dtype=np.int64):
            halodata[halosnaparray[i]]['RootHead'][haloindexarray[i]]=halodata[istart]['RootHead'][wdata[i]]
            halodata[halosnaparray[i]]['RootHeadSnap'][haloindexarray[i]]=halodata[istart]['RootHeadSnap'][wdata[i]]
            halodata[halosnaparray[i]]['RootHeadIndex'][haloindexarray[i]]=halodata[istart]['RootHeadIndex'][wdata[i]]
        wdata = None
        haloidarray = None
        haloindexarray = None
        halosnaparray = None
    # now go back and find all secondary progenitors and set their root heads
    for istart in snaplist:
        if (numhalos[istart] == 0):
            continue
        # identify all haloes which are not primary progenitors of their descendants, having a descendant rank >0
        wdata = np.where(halodata[istart]['HeadRank'] > 0)[0]
        numactive = wdata.size
        if (iverbose > 0):
            print('Setting sub branch root heads at ', istart, 'halos', numhalos[istart], 'active', numactive)
            sys.stdout.flush()
        if (numactive == 0):
            continue
        # sort this list based on descendant ranking
        sortedranking = np.argsort(halodata[istart]['HeadRank'][wdata])
        rankedhalos = halodata[istart]['ID'][wdata[sortedranking]]
        rankedhaloindex = np.array(rankedhalos % TEMPORALHALOIDVAL - 1, dtype=np.int64)
        wdata = None
        maindescen = np.array([tree[istart]['Descen'][index][0] for index in rankedhaloindex], dtype=np.int64)
        maindescenindex = np.array(maindescen % TEMPORALHALOIDVAL - 1, dtype=np.int64)
        maindescensnap = np.array((maindescen - maindescenindex - np.int64(1)) / TEMPORALHALOIDVAL, dtype=np.int32)
        if (ireverseorder):
            maindescensnap = numsnaps - 1 - maindescensnap
        # for each of these haloes, set the head and use the root head information and root snap and set all the information
        # long its branch
        for i in range(numactive):
            # store the root head
            # now set the head of these objects
            halosnap = istart
            haloid = rankedhalos[i]
            haloindex = rankedhaloindex[i]
            # increase the number of progenitors of this descendant
            roothead = halodata[maindescensnap[i]]['RootHead'][maindescenindex[i]]
            rootsnap = halodata[maindescensnap[i]]['RootHeadSnap'][maindescenindex[i]]
            rootindex = halodata[maindescensnap[i]]['RootHeadIndex'][maindescenindex[i]]
            # now set the root head for all the progenitors of this object
            while (True):
                halodata[halosnap]['RootHead'][haloindex] = roothead
                halodata[halosnap]['RootHeadSnap'][haloindex] = rootsnap
                halodata[halosnap]['RootHeadIndex'][haloindex] = rootindex
                if (haloid == halodata[halosnap]['Tail'][haloindex]):
                    break
                haloid = halodata[halosnap]['Tail'][haloindex]
                tmphaloindex = halodata[halosnap]['TailIndex'][haloindex]
                halosnap = halodata[halosnap]['TailSnap'][haloindex]
                haloindex = tmphaloindex
        rankedhalos = None
        rankedhaloindex = None
        maindescen = None
        maindescenindex = None
        maindescensnaporder = None

    print("Done building", time.process_time()-totstart)


def GetProgenLength(halodata, haloindex, halosnap, haloid, atime, TEMPORALHALOIDVAL, endreftime=-1):
    """
    Get the length of a halo's progenitors
    """
    proglen = 1
    progid = halodata[halosnap]["Tail"][haloindex]
    progsnap = halodata[halosnap]["TailSnap"][haloindex]
    progindex = int(progid % TEMPORALHALOIDVAL-1)
    while (progid != haloid):
        if (atime[progsnap] <= endreftime):
            break
        proglen += 1
        haloid = progid
        halosnap = progsnap
        haloindex = progindex
        progid = halodata[halosnap]["Tail"][haloindex]
        progsnap = halodata[halosnap]["TailSnap"][haloindex]
        progindex = int(progid % TEMPORALHALOIDVAL-1)
    return proglen


def IdentifyMergers(numsnaps, tree, numhalos, halodata, boxsize, hval, atime, MERGERMLIM=0.1, RADINFAC=1.2, RADOUTFAC=1.5, NPARTCUT=100, TEMPORALHALOIDVAL=1000000000000, iverbose=1, pos_tree=[]):
    """
    Using head/tail info in halodata dictionary identify mergers based on distance and mass ratios
    #todo still testing

    """

    #Get the snapshot offset if present in the header information
    if("HaloID_snapshot_offset" in tree["Header"]):
        snapshotoffset = tree["Header"]["HaloID_snapshot_offset"]
    else:
        snapshotoffset = 0

    for j in range(snapshotoffset,snapshotoffset+numsnaps):
        # store id and snap and mass of last major merger and while we're at it, store number of major mergers
        halodata[j]["LastMerger"] = np.ones(numhalos[j], dtype=np.int64)*-1
        halodata[j]["LastMergerRatio"] = np.ones(
            numhalos[j], dtype=np.float64)*-1
        halodata[j]["LastMergerSnap"] = np.zeros(numhalos[j], dtype=np.uint32)
        halodata[j]["LastMergerDeltaSnap"] = np.zeros(
            numhalos[j], dtype=np.uint32)
        #halodata[j]["NumMergers"] = np.zeros(numhalos[j], dtype = np.uint32)
    # built KD tree to quickly search for near neighbours
    if (len(pos_tree) == 0):
        pos = [[]for j in range(snapshotoffset,snapshotoffset+numsnaps)]
        pos_tree = [[]for j in range(snapshotoffset,snapshotoffset+numsnaps)]
        start = time.process_time()
        if (iverbose):
            print("tree build")
        for j in range(snapshotoffset,snapshotoffset+numsnaps):
            if (numhalos[j] > 0):
                boxval = boxsize*atime[j]/hval
                pos[j] = np.transpose(np.asarray(
                    [halodata[j]["Xc"], halodata[j]["Yc"], halodata[j]["Zc"]]))
                pos_tree[j] = spatial.cKDTree(pos[j], boxsize=boxval)
        if (iverbose):
            print("done ", time.process_time()-start)
    # else assume tree has been passed
    for j in range(snapshotoffset,snapshotoffset+numsnaps):
        if (numhalos[j] == 0):
            continue
        # at snapshot look at all haloes that have not had a major merger set
        # note that only care about objects with certain number of particles
        partcutwdata = np.where(halodata[j]["npart"] >= NPARTCUT)
        mergercut = np.where(halodata[j]["LastMergerRatio"][partcutwdata] < 0)
        hids = np.asarray(halodata[j]["ID"][partcutwdata]
                          [mergercut], dtype=np.uint64)
        start = time.process_time()
        if (iverbose):
            print("Processing ", len(hids))
        if (len(hids) == 0):
            continue

        for hidval in hids:
            # now for each object get the main progenitor
            haloid = np.uint64(hidval)
            haloindex = int(haloid % TEMPORALHALOIDVAL-1)
            halosnap = j
            originalhaloid = haloid
            progid = halodata[halosnap]["Tail"][haloindex]
            progsnap = halodata[halosnap]["TailSnap"][haloindex]
            progindex = int(progid % TEMPORALHALOIDVAL-1)
            numprog = tree[halosnap]["Num_progen"][haloindex]
            # if object has no progenitor set LastMergerRatio to 0 and LastMerger to 0
            if (numprog == 0):
                halodata[halosnap]["LastMerger"][haloindex] = 0
                halodata[halosnap]["LastMergerRatio"][haloindex] = 0
                continue
            # print "starting halos ", j, hidval
            # halo has main branch which we can wander on
            # while object is not its own progenitor move along tree to see how many major mergers it had across its history
            while (True):
                # now for each progenitor, lets find any nearby objects within a given mass/vmax interval
                posval = [halodata[progsnap]["Xc"][progindex], halodata[progsnap]
                          ["Yc"][progindex], halodata[progsnap]["Zc"][progindex]]
                radval = RADINFAC*halodata[progsnap]["R_200crit"][progindex]
                # get neighbour list within RADINFAC sorted by mass with most massive first
                NNlist = pos_tree[progsnap].query_ball_point(posval, radval)
                NNlist = [NNlist[ij] for ij in np.argsort(
                    halodata[progsnap]["Mass_tot"][NNlist])[::-1]]
                # store boxval for periodic correction
                boxval = boxsize*atime[progsnap]/hval
                # now if list contains some objects, lets see if the velocity vectors are moving towards each other and mass/vmax ratios are okay
                if (len(NNlist) > 0):
                    for NN in NNlist:
                        if (NN != progindex):
                            mratio = halodata[progsnap]["Mass_tot"][NN] / \
                                halodata[progsnap]["Mass_tot"][progindex]
                            vratio = halodata[progsnap]["Vmax"][NN] / \
                                halodata[progsnap]["Vmax"][progindex]
                            # merger ratio is for object being larger of the two involved in merger
                            if (mratio > MERGERMLIM and mratio < 1.0):
                                posvalrel = [halodata[progsnap]["Xc"][progindex]-halodata[progsnap]["Xc"][NN], halodata[progsnap]["Yc"]
                                             [progindex]-halodata[progsnap]["Yc"][NN], halodata[progsnap]["Zc"][progindex]-halodata[progsnap]["Zc"][NN]]
                                for ij in range(3):
                                    if posvalrel[ij] < -0.5*boxval:
                                        posvalrel[ij] += boxval
                                    elif posvalrel[ij] > 0.5*boxval:
                                        posvalrel[ij] -= boxval
                                velvalrel = [halodata[progsnap]["VXc"][progindex]-halodata[progsnap]["VXc"][NN], halodata[progsnap]["VYc"]
                                             [progindex]-halodata[progsnap]["VYc"][NN], halodata[progsnap]["VZc"][progindex]-halodata[progsnap]["VZc"][NN]]
                                radvelval = np.dot(
                                    posvalrel, velvalrel)/np.linalg.norm(posvalrel)
                                if (radvelval < 0):
                                    #merger is happending
                                    # print "merger happening ", progsnap, NN

                                    # question of whether should move down the tree till merger no longer happening and define that as the start
                                    # this could also set the length of the merger
                                    # lets move along the tree of the infalling neighbour still it object is past the some factor of progenitor virial radius
                                    starthaloindex = progindex
                                    starthaloid = progid
                                    starthalosnap = progsnap
                                    startmergerindex = NN
                                    startmergerid = halodata[progsnap]["ID"][NN]
                                    startmergersnap = progsnap
                                    mergerstartindex = starthaloindex
                                    mergerstartid = starthaloid
                                    mergerstartsnap = starthalosnap
                                    while (tree[starthalosnap]["Num_progen"][starthaloindex] > 0 and tree[startmergersnap]["Num_progen"][startmergerindex] > 0):
                                        posvalrel = [halodata[starthalosnap]["Xc"][starthaloindex]-halodata[startmergersnap]["Xc"][startmergerindex], halodata[starthalosnap]["Yc"][starthaloindex] -
                                                     halodata[startmergersnap]["Yc"][startmergerindex], halodata[starthalosnap]["Zc"][starthaloindex]-halodata[startmergersnap]["Zc"][startmergerindex]]
                                        boxval = boxsize * \
                                            atime[starthalosnap]/hval
                                        for ij in range(3):
                                            if posvalrel[ij] < -0.5*boxval:
                                                posvalrel[ij] += boxval
                                            elif posvalrel[ij] > 0.5*boxval:
                                                posvalrel[ij] -= boxval
                                        radval = np.linalg.norm(
                                            posvalrel)/halodata[starthalosnap]["R_200crit"][starthaloindex]
                                        mratio = halodata[startmergersnap]["Mass_tot"][startmergerindex] / \
                                            halodata[starthalosnap]["Mass_tot"][starthaloindex]

                                        # as moving back if halo now outside or too small, stop search and define this as start of merger
                                        if (radval > RADOUTFAC or mratio < MERGERMLIM):
                                            mergerstartindex = starthaloindex
                                            mergerstartid = starthaloid
                                            mergerstartsnap = starthalosnap
                                            break

                                        # move to next progenitors
                                        nextidval = halodata[starthalosnap]["Tail"][starthaloindex]
                                        nextsnapval = halodata[starthalosnap]["TailSnap"][starthaloindex]
                                        nextindexval = int(
                                            nextidval % TEMPORALHALOIDVAL-1)
                                        starthaloid = nextidval
                                        starthalosnap = nextsnapval
                                        starthaloindex = nextindexval

                                        nextidval = halodata[startmergersnap]["Tail"][startmergerindex]
                                        nextsnapval = halodata[startmergersnap]["TailSnap"][startmergerindex]
                                        nextindexval = int(
                                            nextidval % TEMPORALHALOIDVAL-1)
                                        startmergerid = nextidval
                                        startmergersnap = nextsnapval
                                        startmergerindex = nextindexval
                                    # store timescale of merger
                                    deltamergertime = (
                                        mergerstartsnap-progsnap)
                                    # set this as the merger for all halos from this point onwards till reach head or halo with non-zero merger
                                    merginghaloindex = mergerstartindex
                                    merginghaloid = mergerstartid
                                    merginghalosnap = mergerstartsnap
                                    oldmerginghaloid = merginghaloid
                                    # print "Merger found ", progsnap, mergerstartsnap, halodata[progsnap]["Mass_tot"][NN]/halodata[progsnap]["Mass_tot"][progindex],
                                    # print halodata[startmergersnap]["Mass_tot"][startmergerindex]/halodata[starthalosnap]["Mass_tot"][starthaloindex]
                                    # now set merger time for all later haloes unless an new merger has happened
                                    while (oldmerginghaloid != halodata[progsnap]["RootHead"][progindex] and halodata[merginghalosnap]["LastMergerRatio"][merginghaloindex] < 0):
                                        halodata[merginghalosnap]["LastMerger"][merginghaloindex] = halodata[progsnap]["ID"][NN]
                                        halodata[merginghalosnap]["LastMergerRatio"][merginghaloindex] = halodata[
                                            progsnap]["Mass_tot"][NN]/halodata[progsnap]["Mass_tot"][progindex]
                                        halodata[merginghalosnap]["LastMergerSnap"][merginghaloindex] = progsnap
                                        halodata[merginghalosnap]["LastMergerDeltaSnap"][merginghaloindex] = deltamergertime

                                        oldmerginghaloid = merginghaloid
                                        mergingnextid = halodata[merginghalosnap]["Head"][merginghaloindex]
                                        mergingnextsnap = halodata[merginghalosnap]["HeadSnap"][merginghaloindex]
                                        mergingnextindex = int(
                                            mergingnextid % TEMPORALHALOIDVAL-1)
                                        merginghaloindex = mergingnextindex
                                        merginghaloid = mergingnextid
                                        merginghalosnap = mergingnextsnap

                # move to next step
                if (haloid == progid):
                    oldhaloid = haloid
                    currentsnap = halosnap
                    currentindex = haloindex
                    currentid = haloid
                    while (oldhaloid != halodata[progsnap]["RootHead"][progindex] and halodata[currentsnap]["LastMergerRatio"][currentindex] < 0):
                        halodata[currentsnap]["LastMerger"][currentindex] = 0
                        halodata[currentsnap]["LastMergerRatio"][currentindex] = 0
                        nextid = halodata[currentsnap]["Head"][currentindex]
                        nextsnap = halodata[currentsnap]["HeadSnap"][currentindex]
                        nextindex = int(nextid % TEMPORALHALOIDVAL-1)
                        oldhaloid = currentid
                        currentsnap = nextsnap
                        currentid = nextid
                        currentindex = nextindex
                    break
                haloid = progid
                haloindex = progindex
                halosnap = progsnap
                progid = halodata[halosnap]["Tail"][haloindex]
                progsnap = halodata[halosnap]["TailSnap"][haloindex]
                progindex = int(progid % TEMPORALHALOIDVAL-1)
                numprog = tree[halosnap]["Num_progen"][haloindex]
                # if at end of line then move up and set last major merger to 0
        if (iverbose):
            print("Done snap", j, time.process_time()-start)


def generate_sublinks(numhalos, halodata, iverbose=0):
    """
    generate sublinks for specific time slice
    """
    if (numhalos == 0):
        return
    # get hosts and also all subhalos
    hosts = np.where(halodata['hostHaloID'] == -1)[0]
    hostsids = halodata['ID'][hosts]
    hostsnumsubs = np.array(halodata['numSubStruct'][hosts], dtype=np.int64)
    subhalos = np.where(np.in1d(halodata['hostHaloID'], hostsids))[0]
    # order by host halo ID
    subhalos = subhalos[np.argsort(halodata['hostHaloID'][subhalos])]
    # then init Next/ Previous to point to itself

    # generate offset based on hostnumsubs
    hostssuboffset = np.zeros(len(hosts), dtype=np.int64)
    hostssuboffset[1:] = np.cumsum(hostsnumsubs[:-1])

    if (iverbose):
        print(len(hosts), len(subhalos))
    # now for all hosts, start iterating
    for ihalo in range(len(hosts)):
        if (hostsnumsubs[ihalo] == 0):
            continue
        prevsub = halodata['ID'][hosts[ihalo]]
        halodata['PreviousSubhalo'][hosts[ihalo]] = prevsub
        nextsub = halodata['ID'][subhalos[hostssuboffset[ihalo]]]
        halodata['NextSubhalo'][hosts[ihalo]] = nextsub
        for j in range(hostsnumsubs[ihalo]-1):
            halodata['PreviousSubhalo'][subhalos[hostssuboffset[ihalo]+j]] = prevsub
            nextsub = halodata['ID'][subhalos[hostssuboffset[ihalo]+j+1]]
            halodata['NextSubhalo'][subhalos[hostssuboffset[ihalo]+j]] = nextsub
            prevsub = halodata['ID'][subhalos[hostssuboffset[ihalo]+j]]
        halodata['PreviousSubhalo'][subhalos[hostssuboffset[ihalo] +
                                             hostsnumsubs[ihalo]-1]] = prevsub
        halodata['NextSubhalo'][subhalos[hostssuboffset[ihalo]+hostsnumsubs[ihalo]-1]
                                ] = halodata['ID'][subhalos[hostssuboffset[ihalo]+hostsnumsubs[ihalo]-1]]


def GenerateSubhaloLinks(numsnaps, numhalos, halodata, TEMPORALHALOIDVAL=1000000000000, iverbose=0, iparallel=0):
    """
    This code generates a quick way of moving across a halo's subhalo list

    The code is passed
    - the number of snapshots,
    - an array of the number of haloes per snapshot,
    - the halodata dictionary structure which must contain the halo merger tree based keys, Head, RootHead, etc, and mass, phase-space positions of haloes,
    and other desired properties
    """
    #Get the snapshot offset if present in the header information
    if("HaloID_snapshot_offset" in halodata["Header"]):
        snapshotoffset = halodata["Header"]["HaloID_snapshot_offset"]
    else:
        snapshotoffset = 0


    for j in range(snapshotoffset,snapshotoffset+numsnaps):
        # store id and snap and mass of last major merger and while we're at it, store number of major mergers
        halodata[j]["NextSubhalo"] = copy.deepcopy(halodata[j]["ID"])
        halodata[j]["PreviousSubhalo"] = copy.deepcopy(halodata[j]["ID"])
    # iterate over all host halos and set their subhalo links
    start = time.process_time()
    nthreads = 1
    if (iparallel):
        manager = mp.Manager()
        nthreads = int(min(mp.cpu_count(), numsnaps))
        print("Number of threads is ", nthreads)
    for j in range(0, numsnaps, nthreads):
        start2 = time.process_time()
        if (iparallel):
            activenthreads = nthreads
            if (numsnaps-1-j < activenthreads):
                activenthreads = numsnaps-1-j
            processes = [mp.Process(target=generate_sublinks, args=(
                numhalos[j+k], halodata[j+k], iverbose)) for k in range(activenthreads)]
            for p in processes:
                p.start()
            for p in processes:
                p.join()
            if (iverbose):
                print("Done snaps", j, "to", j+nthreads, time.process_time()-start2)
                sys.stdout.flush()

        else:
            generate_sublinks(numhalos[j], halodata[j], iverbose)
            if (iverbose):
                print("Done snap", j, time.process_time()-start2)
                sys.stdout.flush()
    print("Done subhalolinks ", time.process_time()-start)
    sys.stdout.flush()


def GenerateProgenitorLinks(numsnaps, numhalos, halodata, ireversesnaporder=False,
                            nsnapsearch=4, TEMPORALHALOIDVAL=1000000000000, iverbose=0):
    """
    This code generates a quick way of moving across a halo's progenitor list storing a the next/previous progenitor

    The code is passed
    - the number of snapshots,
    - an array of the number of haloes per snapshot,
    - the halodata dictionary structure which must contain the halo merger tree based keys, Head, RootHead, etc, and mass, phase-space positions of haloes,
    and other desired properties
    """
    if (nsnapsearch >= numsnaps-1):
        nsnapsearch = numsnaps-1
        print("Warning, number of snaps < search size, reducing search size to numsnaps-1 = ", nsnapsearch)

    #Get the snapshot offset if present in the header information
    if("HaloID_snapshot_offset" in halodata["Header"]):
        snapshotoffset = halodata["Header"]["HaloID_snapshot_offset"]
    else:
        snapshotoffset = 0

    for j in range(snapshotoffset,snapshotoffset+numsnaps):
        # store id and snap and mass of last major merger and while we're at it, store number of major mergers
        halodata[j]["LeftTail"] = copy.deepcopy(halodata[j]["ID"])
        halodata[j]["RightTail"] = copy.deepcopy(halodata[j]["ID"])
        # alias the data
        halodata[j]["PreviousProgenitor"] = halodata[j]["LeftTail"]
        halodata[j]["NextProgenitor"] = halodata[j]["RightTail"]
    # move backward in time and identify all unique heads
    start = time.process_time()
    if (ireversesnaporder):
        snaplist = range(1, numsnaps)
    else:
        snaplist = range(numsnaps-2, -1, -1)
    for j in snaplist:
        start2 = time.process_time()
        if (numhalos[j] == 0):
            continue
        if (numhalos[j+1] == 0):
            continue
        # get hosts and also all subhalos
        heads = halodata[j+1]['ID']
        # for these IDs identify all halos with this as their head
        if (ireversesnaporder):
            snaplist2 = np.arange(
                j, min(j+nsnapsearch, numsnaps), dtype=np.int32)
        else:
            snaplist2 = np.arange(
                j, max(j-nsnapsearch, -1), -1, dtype=np.int32)
        progens = []
        progenssnaps = []
        progensheads = []
        progensids = []
        for k in snaplist2:
            wdata = np.where(np.in1d(halodata[k]['Head'], heads))
            if (len(wdata[0]) == 0):
                continue
            progens.append(wdata[0])
            progenssnaps.append(np.ones(len(progens[-1]))*k)
            progensheads.append(halodata[k]['Head'][wdata])
            progensids.append(halodata[k]['ID'][wdata])
        # flatten and then reorder to group stuff by head
        progens = np.array(np.concatenate(progens), dtype=np.int64)
        progenssnaps = np.array(np.concatenate(progenssnaps), dtype=np.int32)
        progensheads = np.array(np.concatenate(progensheads), dtype=np.int64)
        progensids = np.array(np.concatenate(progensids), dtype=np.int64)
        nprogs = len(progens)
        if (nprogs < 2):
            continue
        idx = np.argsort(progensheads)
        progens, progenssnaps, progensheads, progensids = progens[
            idx], progenssnaps[idx], progensheads[idx], progensids[idx]
        # now move along the length of the progen array to set up current and previous
        activehead = progensheads[0]
        #prevprog, nextprog = -1, progensids[0]
        prevprog, nextprog = progensids[0], progensids[0]
        index, snap = progens[0], progenssnaps[0]
        #halodata[snap]['LeftTail'][index] = -1
        halodata[snap]['LeftTail'][index] = halodata[snap]['ID'][index]
        for iprog in range(1, nprogs-1):
            index, snap = progens[iprog], progenssnaps[iprog]
            nextindex, nextsnap = progens[iprog+1], progenssnaps[iprog+1]
            previndex, prevsnap = progens[iprog-1], progenssnaps[iprog-1]
            curhead = progensheads[iprog]
            if (curhead != activehead):

                halodata[snap]['LeftTail'][index] = halodata[snap]['ID'][index]
                halodata[prevsnap]['RightTail'][previndex] = halodata[prevsnap]['ID'][previndex]
                prevprog = halodata[snap]['ID'][index]

                activehead = curhead
            else:
                nextprog = progensids[iprog]
                halodata[snap]['LeftTail'][index] = prevprog
                halodata[prevsnap]['RightTail'][previndex] = nextprog
                prevprog = progensids[iprog]
        curhead = progensheads[-1]
        index, snap = progens[-1], progenssnaps[-1]
        halodata[snap]['RightTail'][index] = halodata[snap]['ID'][index]
        if (iverbose):
            print("Done snap", j, time.process_time()-start2)
            sys.stdout.flush()
    print("Done progenitor links ", time.process_time()-start)
    sys.stdout.flush()


def GenerateForest(numsnaps, numhalos, halodata, atime, nsnapsearch=4,
                   ireversesnaporder=False, TEMPORALHALOIDVAL=1000000000000, iverbose=2,
                   icheckforest=False,
                   interactiontime=2, ispatialintflag=False, pos_tree=[], cosmo=dict()):
    """
    This code traces all root heads back in time identifying all interacting haloes and bundles them together into the same forest id
    The idea is to have in the halodata dictionary an associated unique forest id for all related (sub)haloes. The code also allows
    for some cleaning of the forest, specifically if a (sub)halo is only interacting for some small fraction of time, then it is not
    assigned to the forest. This can limit the size of a forest, which could otherwise become the entire halo catalog.

    Parameters
    ----------
    numsnaps : numpy.int32
        the number of snapshots
    numhalos : array
        array of the number of haloes per snapshot.
    halodata : dict
        the halodata dictionary structure which must contain the halo merger tree based keys (Head, RootHead), etc.
    atime : array
        an array of scale factors

    Optional Parameters
    -------------------
    ireversesnaporder : bool
        Whether first snap is at [0] (False) or last snap is at [0] (True)
    TEMPORALHALOIDVAL : numpy.int64
        Temporal ID value that makes Halo IDs temporally unique, adding a snapshot num* this value.
        Allows one to quickly parse a Halo ID to determine the snapshot it exists at and its index.
    iverbose : int
        verbosity of function (0, minimal, 1, verbose, 2 chatterbox)
    icheckforest : bool
        run final check on forest
    interactiontime : int
        Optional functionality not implemented yet. Allows forest to be split if connections do not span
        more than this number of snapshots
    ispatialintflag : bool
        Flag indicating whether spatial information should be used to join forests. This requires cosmological information
    pos_tree : scikit.spatial.cKDTree
        Optional functionality not implemented yet. Allows forests to be joined if haloes
        are spatially close.
    cosmo : dict
        dictionary which has cosmological information such as box size, hval, Omega_m

    Returns
    -------
    ForestSize : numpy.array
        Update the halodata dictionary with ForestID information and also returns the size of
        the forests

    """

        #Get the snapshot offset if present in the header information
    if("HaloID_snapshot_offset" in halodata["Header"]):
        snapshotoffset = halodata["Header"]["HaloID_snapshot_offset"]
    else:
        snapshotoffset = 0

    # initialize the dictionaries
    for j in range(snapshotoffset,snapshotoffset+numsnaps):
        # store id and snap and mass of last major merger and while we're at it, store number of major mergers
        halodata[j]["ForestID"] = np.ones(numhalos[j], dtype=np.int64)*-1
        halodata[j]["ForestLevel"] = np.ones(numhalos[j], dtype=np.int32)*-1
    # built KD tree to quickly search for near neighbours. only build if not passed.
    if (ispatialintflag):
        start = time.process_time()
        boxsize = cosmo['BoxSize']
        hval = cosmo['Hubble_param']
        if (len(pos_tree) == 0):
            pos = [[]for j in range(snapshotoffset,snapshotoffset+numsnaps)]
            pos_tree = [[]for j in range(snapshotoffset,snapshotoffset+numsnaps)]
            start = time.process_time()
            if (iverbose):
                print("KD tree build")
                sys.stdout.flush()
            for j in range(snapshotoffset,snapshotoffset+numsnaps):
                if (numhalos[j] > 0):
                    boxval = boxsize*atime[j]/hval
                    pos[j] = np.transpose(np.asarray(
                        [halodata[j]["Xc"], halodata[j]["Yc"], halodata[j]["Zc"]]))
                    pos_tree[j] = spatial.cKDTree(pos[j], boxsize=boxval)
            if (iverbose):
                print("done ", time.process_time()-start)
                sys.stdout.flush()

    # now start marching backwards in time from root heads
    # identifying all subhaloes that have every been subhaloes for long enough
    # and all progenitors and group them together into the same forest id
    forestidval = 1
    start = time.process_time()
    # for j in range(snapshotoffset,snapshotoffset+numsnaps):
    # set the direction of how the data will be processed
    if (ireversesnaporder):
        snaplist = np.arange(0, numsnaps, dtype=np.int32)
    else:
        snaplist = np.arange(numsnaps-1, -1, -1)
    # first pass assigning forests based on FOF and subs
    offset = 0
    start2 = time.process_time()
    print('starting first pass in producing forest ids using', nsnapsearch,
          'snapshots being serached and', TEMPORALHALOIDVAL, 'defining temporal id')
    sys.stdout.flush()
    for j in snaplist:
        if (numhalos[j] == 0):
            continue
        hosts = np.where(halodata[j]['hostHaloID'] == -1)
        halodata[j]['ForestID'][hosts] = halodata[j]['ID'][hosts]
        subs = np.where(halodata[j]['hostHaloID'] != -1)[0]
        if (subs.size == 0):
            continue
        halodata[j]['ForestID'][subs] = halodata[j]['hostHaloID'][subs]

    # get initial size of each forest where forests are grouped by halo+subhalo relation
    ForestIDs, ForestSize = np.unique(np.concatenate(
        [halodata[i]['ForestID'] for i in range(snapshotoffset,snapshotoffset+numsnaps)]), return_counts=True)
    numforests = len(ForestIDs)
    maxforest = np.max(ForestSize)
    print('finished first pass', time.process_time()-start2,
          'have ', numforests, 'initial forests',
          'with largest forest containing ', maxforest, '(sub)halos')
    sys.stdout.flush()

    #ForestSizeStats = dict(zip(ForestIDs, ForestSize))
    #store the a map of forest ids that will updated
    ForestMap = dict(zip(ForestIDs, ForestIDs))
    # free memory
    ForestIDs = ForestSize = None

    # now proceed to find new mappings
    start1 = time.process_time()

    #above handles primary progenitor/substructure but now also need to handle secondary progenitors
    #particularly of objects with no primary progenitor
    numloops = 0
    while (True):
        newforests = 0
        start2 = time.process_time()
        if (iverbose):
            print('walking forward in time to identify forest id mappings')
            sys.stdout.flush()
        snaplist = np.arange(numsnaps,dtype=np.int32)
        if (ireversesnaporder):
            snaplist = snaplist[::-1]
        for j in snaplist:
            if (numhalos[j] == 0):
                continue
            start3 = time.process_time()
            if (ireversesnaporder):
                endsnapsearch = max(0, j-nsnapsearch-1)
                snaplist2 = np.arange(j-1, endsnapsearch, -1, dtype=np.int32)
            else:
                endsnapsearch = min(numsnaps, j+nsnapsearch+1)
                snaplist2 = np.arange(j+1, endsnapsearch, dtype=np.int32)
            incforests = 0
            for k in snaplist2:
                if (numhalos[k] == 0):
                    continue
                descens = None
                #find all descendants of objects that have descendants at snapshot k
                tailindexarray = np.where(np.int32(halodata[j]['Head']/TEMPORALHALOIDVAL) == k)[0]
                if (len(tailindexarray) == 0):
                    continue
                descens = np.int64(halodata[j]['Head'][tailindexarray] % TEMPORALHALOIDVAL-1)
                if (numloops >= 1):
                    wdata = np.where(halodata[k]['ForestID'][descens] != halodata[j]['ForestID'][tailindexarray])
                    if (len(wdata[0]) == 0):
                        continue
                    tailindexarray = tailindexarray[wdata]
                    descens = np.int64(halodata[j]['Head'][tailindexarray] % TEMPORALHALOIDVAL-1)
                    wdata = None
                #print('snap',j,'to snap',k,'refforest active', tailindexarray.size, 'curforest active', descens.size)
                # process snap to update forest id map
                for icount in range(tailindexarray.size):
                    itail = tailindexarray[icount]
                    idescen = descens[icount]
                    curforest = halodata[k]['ForestID'][idescen]
                    refforest = halodata[j]['ForestID'][itail]
                    # it is possible that after updating can have the descedants forest id match its progenitor forest id so do nothing if this is the case
                    if (ForestMap[curforest] == ForestMap[refforest]):
                        continue
                    # if ref forest is smaller update the mapping
                    if (ForestMap[curforest] > ForestMap[refforest]):
                        ForestMap[curforest] = ForestMap[refforest]
                        newforests += 1
                        incforests += 1
                    else :
                        ForestMap[refforest] = ForestMap[curforest]
                        newforests += 1
                        incforests += 1
        if (iverbose):
            print('done walking forward, found  ', newforests, ' new forest links at ',
                  numloops, ' loop in a time of ', time.process_time()-start2)
            sys.stdout.flush()
        # update forest ids using map
        start2 = time.process_time()
        for j in range(snapshotoffset,snapshotoffset+numsnaps):
            if (numhalos[j] == 0):
                continue
            for ihalo in range(numhalos[j]):
                halodata[j]['ForestID'][ihalo]=ForestMap[halodata[j]['ForestID'][ihalo]]
        if (newforests == 0):
            break
        if (iverbose):
            print('Finished remapping in', time.process_time()-start2)
            sys.stdout.flush()
        numloops += 1

    print('Done forests in %d in a time of %f' %
          (numloops, time.process_time()-start1))
    sys.stdout.flush()

    # get the size of each forest
    ForestIDs, ForestSize = np.unique(np.concatenate(
        [halodata[i]['ForestID'] for i in range(snapshotoffset,snapshotoffset+numsnaps)]), return_counts=True)
    numforests = ForestIDs.size
    maxforest = np.max(ForestSize)
    foreststats = np.percentile(ForestSize,[16.0,50.0,84.0,95.0,99.0,99.5])
    print('Forest consists of ', numforests, 'with largest', maxforest, 'forest size stats', foreststats)

    ForestSizeStats = dict()
    ForestSizeStats['ForestIDs'] = ForestIDs
    ForestSizeStats['ForestSizes'] = ForestSize
    ForestSizeStats['AllSnaps'] = dict(zip(ForestIDs, ForestSize))
    ForestSizeStats['Number_of_forests'] = numforests
    ForestSizeStats['Max_forest_size'] = maxforest
    ForestSizeStats['Max_forest_fof_groups_size'] = maxforest
    ForestSizeStats['Max_forest_ID'] = ForestIDs[np.argmax(ForestSize)]
    ForestSizeStats['Snapshots'] = {
        'Number_of_active_forests': dict(),
        'Max_forest_size': dict(),
        'Max_forest_fof_groups_size': dict(),
        'Max_forest_ID': dict(),
        'Max_forest_fof_groups_ID': dict(),
        'Number_of_halos_in_forests' : dict(),
        'Number_of_fof_groups_in_forests' : dict(),
        'Active_forests' : dict(),
        }
    #analyse snapshot by snapshot
    cumcounts = np.zeros(numforests, dtype=np.int64)
    cumcounts_fof = np.zeros(numforests, dtype=np.int64)
    for i in range(snapshotoffset,snapshotoffset+numsnaps):
        snkey = 'Snap_%03d' % i
        for key in ForestSizeStats['Snapshots'].keys():
            ForestSizeStats['Snapshots'][key][snkey] = 0
        ForestSizeStats['Snapshots']['Active_forests'][snkey] = None
        ForestSizeStats['Snapshots']['Number_of_halos_in_forests'][snkey] = np.zeros(numforests, dtype=np.int64)
        ForestSizeStats['Snapshots']['Number_of_fof_groups_in_forests'][snkey] = np.zeros(numforests, dtype=np.int64)

        if (numhalos[i] == 0): continue
        activeforest, counts = np.unique(halodata[i]['ForestID'], return_counts=True)
        ForestSizeStats['Snapshots']['Number_of_active_forests'][snkey] = activeforest.size
        ForestSizeStats['Snapshots']['Active_forests'][snkey] = np.array(activeforest)
        wdata = np.where(np.in1d(ForestIDs, activeforest))[0]
        cumcounts[wdata] += counts
        ForestSizeStats['Snapshots']['Number_of_halos_in_forests'][snkey][wdata] = counts
        ForestSizeStats['Snapshots']['Max_forest_size'][snkey] = np.max(counts)
        ForestSizeStats['Snapshots']['Max_forest_ID'][snkey] = ForestIDs[np.argmax(counts)]
        wfof = np.where(halodata[i]['hostHaloID'] == -1)[0]
        if (wfof.size == 0): continue
        activeforest, counts = np.unique(halodata[i]['ForestID'][wfof], return_counts=True)
        wdata = np.where(np.in1d(ForestIDs, activeforest))[0]
        cumcounts_fof[wdata] += counts
        ForestSizeStats['Snapshots']['Number_of_fof_groups_in_forests'][snkey][wdata] = counts
        ForestSizeStats['Snapshots']['Max_forest_fof_groups_size'][snkey] = np.max(counts)
        ForestSizeStats['Snapshots']['Max_forest_fof_groups_ID'][snkey] = ForestIDs[np.argmax(counts)]
    #some some final cumulative stats
    ForestSizeStats['Max_forest_fof_groups_size'] = np.max(cumcounts_fof)
    ForestSizeStats['Max_forest_fof_groups_ID'] = ForestIDs[np.argmax(cumcounts_fof)]

    start2 = time.process_time()
    if (icheckforest):
        # first identify all subhalos and see if any have subhalo connections with different than their host
        if (ireversesnaporder):
            snaplist = np.arange(0, numsnaps, dtype=np.int32)
        else:
            snaplist = np.arange(numsnaps-1, -1, -1)
        for j in snaplist:
            if (numhalos[j] == 0):
                 continue
            subs = np.where(halodata[j]['hostHaloID'] != -1)[0]
            nomatchsubs = 0
            if (subs.size==0): continue
            hosts = np.array(halodata[j]['hostHaloID'][subs] % 1000000000000 - 1, dtype=np.int64)
            mismatch = np.where(halodata[j]['ForestID'][subs] != halodata[j]['ForestID'][hosts])[0]
            nomatchsubs += mismatch.size
            if (mismatch.size > 0):
                print('ERROR: snap',j,'nomatch subs',mismatch.size, 'totsubs',subs.size)
        if (nomatchsubs > 0):
            print('ERROR, forest ids show mistmatches between subs and hosts',nomatchsubs)
            print('Returning null and reseting forest ids')
            for j in range(snapshotoffset,snapshotoffset+numsnaps):
                halodata[j]["ForestID"] = np.ones(numhalos[j], dtype=np.int64)*-1
                halodata[j]["ForestLevel"] = np.ones(numhalos[j], dtype=np.int32)*-1
            return []
        numheadtailmismatch = 0
        for j in range(snapshotoffset,snapshotoffset+numsnaps):
            if (numhalos[j] == 0):
                continue
            if (ireversesnaporder):
                endsnapsearch = max(0, j-nsnapsearch-1)
                snaplist2 = np.arange(j-1, endsnapsearch, -1, dtype=np.int32)
            else:
                endsnapsearch = min(numsnaps, j+nsnapsearch+1)
                snaplist2 = np.arange(j+1, endsnapsearch, dtype=np.int32)
            for k in snaplist2:
                if (numhalos[k] == 0):
                    continue
                tailindexarray = np.where(np.int32(halodata[j]['Head']/TEMPORALHALOIDVAL) == k)[0]
                descens = np.int64(halodata[j]['Head'][tailindexarray] % TEMPORALHALOIDVAL-1)
                wdata=np.where(halodata[j]['ForestID'][tailindexarray] != halodata[k]['ForestID'][descens])[0]
                numheadtailmismatch += wdata.size
                if (wdata.size >0):
                    print('ERROR snap', j, 'to', k, 'head tail mismatch number', wdata.size)
        if (numheadtailmismatch > 0):
            print('ERROR, forest ids show mistmatches between head/tail',numheadtailmismatch)
            print('Returning null and reseting forest ids')
            for j in range(snapshotoffset,snapshotoffset+numsnaps):
                halodata[j]["ForestID"] = np.ones(numhalos[j], dtype=np.int64)*-1
                halodata[j]["ForestLevel"] = np.ones(numhalos[j], dtype=np.int32)*-1
            return []

    # then return this
    print("Done generating forest", time.process_time()-start)
    sys.stdout.flush()
    return ForestSizeStats


"""
Adjust halo catalog for period, comoving coords, etc
"""


def AdjustforPeriod(numsnaps, numhalos, halodata, tree, SimInfo={}):
    """
    Map halo positions from 0 to box size

    Parameters
    ----------

    numsnaps : int
        The number of snapshots in the simulation
    numhalos : int
        The number of halos at a given snapshot
    halodata : list of dictionary per snapshot containing the halo properties
        The data structure containing the halo properties

    Other Parameters
    ----------------

    SimInfo : dict
        Dictionary containing the information of the simulation, see Notes below for how this should be set

    Notes
    -----

    If SimInfo is parsed then this is used instead of the dictionary in halodata[snapnum]["SimulationInfo"]. The required structure for SimInfo dictionary is:

    SimInfo = {
    "Comoving":0 if physical or 1 if comoving,
    "Boxsize":comoving boxsize of the simulation box,
    "h_val":reduced hubble parameter (only required if Comoving=0),
    "ScaleFactor":list or array of scalefactors per snapshot (numsnaps long, only required if Comoving=0)
    }

    """

    #Get the snapshot offset if present in the header information
    if("HaloID_snapshot_offset" in tree["Header"]):
        snapshotoffset = tree["Header"]["HaloID_snapshot_offset"]
    else:
        snapshotoffset = 0

    boxval=np.zeros(numsnaps)
    if (all(key in halodata[0].keys() for key in ['UnitInfo','SimulationInfo'])):
        for i in range(snapshotoffset,snapshotoffset+numsnaps):
            if (numhalos[i] == 0):
                continue
            icomove=halodata[i]["UnitInfo"]["Comoving_or_Physical"]
            if (icomove):
                boxval[i] = halodata[i]["SimulationInfo"]["Period"]*halodata[i]["SimulationInfo"]["h_val"]/halodata[i]["SimulationInfo"]["ScaleFactor"]
            else:
                boxval[i] = halodata[i]["SimulationInfo"]["Period"]
    elif (all(key in SimInfo.keys() for key in ['Comoving', 'BoxSize', 'h_val', 'ScaleFactor'])):
        icomove=SimInfo["Comoving"]
        for i in range(snapshotoffset,snapshotoffset+numsnaps):
            if (icomove):
                boxval[i] = SimInfo["Boxsize"]
            else:
                boxval[i] = SimInfo["Boxsize"]*SimInfo["ScaleFactor"]/SimInfo["h_val"]
    else:
        #add a throw of an exception
        print('Missing Info to map positions, doing nothing')
        return

    if("Xc" in halodata[0].keys()):
        distkeys = ['Xc','Yc','Zc']
    elif("Xcminpot" in halodata[0].keys()):
        distkeys = ['Xcminpot','Ycminpot','Zcminpot']
    elif("Xcmbp" in halodata[0].keys()):
        distkeys = ['Xcmbp','Ycmbp','Zcmbp']
    else:
        print("Position dataset not found, please check")

    for i in range(snapshotoffset,snapshotoffset+numsnaps):
        for key in distkeys:
            wdata = np.where(halodata[i][key] < 0)
            halodata[i][key][wdata] += boxval[i]
            wdata = np.where(halodata[i][key] >= boxval[i])
            halodata[i][key][wdata] -= boxval[i]



def AdjustComove(itocomovefromphysnumsnaps, numsnaps, numhalos, atime, halodata, igas=0, istar=0):
    """
    Convert distances to/from physical from/to comoving
    """

    #Get the snapshot offset if present in the header information
    if("HaloID_snapshot_offset" in halodata["Header"]):
        snapshotoffset = halodata["Header"]["HaloID_snapshot_offset"]
    else:
        snapshotoffset = 0

    for i in range(snapshotoffset,snapshotoffset+numsnaps):
        if (numhalos[i] == 0):
            continue
        # converting from physical to comoving
        if (itocomovefromphysnumsnaps == 1):
            fac = float(1.0/atime[i])
        # converting from comoving to physical
        else:
            fac = float(atime[i])
        if (fac == 1):
            continue

        # convert physical distances
        halodata[i]["Xc"] *= fac
        halodata[i]["Yc"] *= fac
        halodata[i]["Zc"] *= fac
        halodata[i]["Xcmbp"] *= fac
        halodata[i]["Ycmbp"] *= fac
        halodata[i]["Zcmbp"] *= fac

        # sizes
        halodata[i]["Rvir"] *= fac
        halodata[i]["R_size"] *= fac
        halodata[i]["R_200mean"] *= fac
        halodata[i]["R_200crit"] *= fac
        halodata[i]["R_BN97"] *= fac
        halodata[i]["Rmax"] *= fac
        halodata[i]["R_HalfMass"] *= fac

        # if gas
        if (igas):
            halodata[i]["Xc_gas"] *= fac
            halodata[i]["Yc_gas"] *= fac
            halodata[i]["Zc_gas"] *= fac
            halodata[i]["R_HalfMass_gas"] *= fac

        # if stars
        if (istar):
            halodata[i]["Xc_star"] *= fac
            halodata[i]["Yc_star"] *= fac
            halodata[i]["Zc_star"] *= fac
            halodata[i]["R_HalfMass_star"] *= fac


"""
Code to use individual snapshot files and merge them together into a full unified hdf file containing information determined from the tree
"""


def WriteUnifiedTreeandHaloCatalog(fname, numsnaps, rawtreedata, numhalos, halodata, atime,
                                   descripdata={'Title': 'Tree and Halo catalog of sim', 'HaloFinder': 'VELOCIraptor', 'Halo_Finder_version': 1.15, 'TreeBuilder': 'TreeFrog', 'Tree_version': 1.1,
                                                'Particle_num_threshold': 20, 'Temporal_linking_length': 1, 'Flag_gas': False, 'Flag_star': False, 'Flag_bh': False,
                                                'Flag_subhalo_links':False, 'Flag_progenitor_links':False, 'Flag_forest_ids':False},
                                   simdata={'Omega_m': 1.0, 'Omega_b': 0., 'Omega_Lambda': 0.,
                                              'Hubble_param': 1.0, 'BoxSize': 1.0, 'Sigma8': 1.0},
                                   unitdata={'UnitLength_in_Mpc': 1.0, 'UnitVelocity_in_kms': 1.0,
                                             'UnitMass_in_Msol': 1.0, 'Flag_physical_comoving': True, 'Flag_hubble_flow': False},
                                   partdata={'Flag_gas': False,
                                             'Flag_star': False, 'Flag_bh': False,
                                             'Particle_mass':{'DarkMatter':-1, 'Gas':-1, 'Star':-1, 'SMBH':-1}},
                                   ibuildheadtail=False,
                                   ibuildforest=False,
                                   idescen=True,
                                   TEMPORALHALOIDVAL=1000000000000,
                                   ireversesnaporder=False):
    """

    produces a unifed HDF5 formatted file containing the full catalog plus information to walk the tree stored in the halo data
    \ref BuildTemporalHeadTail must have been called before otherwise it is called.
    Code produces a file for each snapshot
    The keys are the same as that contained in the halo catalog dictionary with the addition of
    Num_of_snaps, and similar header info contain in the VELOCIraptor hdf files, ie Num_of_groups, Total_num_of_groups

    \todo don't know if I should use multiprocessing here to write files in parallel. IO might not be ideal

    """

    #Get the snapshot offset if present in the header information
    if("HaloID_snapshot_offset" in rawtreedata["Header"]):
        snapshotoffset = rawtreedata["Header"]["HaloID_snapshot_offset"]
    else:
        snapshotoffset = 0

    # check to see in tree data already present in halo catalog
    treekeys = ["RootHead", "RootHeadSnap",
                "Head", "HeadSnap",
                "RootTail", "RootTailSnap",
                "Tail", "TailSnap"
                ]
    # alias names of the tree
    treealiaskeys= ['FinalDescendant', 'FinalDescendantSnap',
                    'Descendant', 'DescendantSnap',
                    'FirstProgenitor', 'FirstProgenitorSnap',
                    'Progenitor', 'ProgenitorSnap'
                    ]
    treealiasnames=dict(zip(treekeys,treealiaskeys))

    if (ibuildheadtail):
        if (set(treekeys).issubset(set(halodata[0].keys())) == False):
            print('building tree')
            if (idescen):
                BuildTemporalHeadTailDescendant(
                    numsnaps, rawtreedata, numhalos, halodata, TEMPORALHALOIDVAL)
            else:
                BuildTemporalHeadTail(
                    numsnaps, rawtreedata, numhalos, halodata, TEMPORALHALOIDVAL)
    if (ibuildforest):
        GenerateForest(numsnaps, numhalos, halodata, atime)
    totnumhalos = sum(numhalos)
    hdffile = h5py.File(fname, 'w')
    headergrp = hdffile.create_group("Header")
    # store useful information such as number of snapshots, halos,
    # cosmology (Omega_m, Omega_b, Hubble_param, Omega_Lambda, Box size)
    # units (Physical [1/0] for physical/comoving flag, length in Mpc, km/s, solar masses, Gravity
    # and TEMPORALHALOIDVAL used to traverse tree information (converting halo ids to haloindex or snapshot), Reverse_order [1/0] for last snap listed first)
    # set the attributes of the header
    headergrp.attrs["NSnaps"] = numsnaps
    headergrp.attrs["Flag_subhalo_links"] = descripdata["Flag_subhalo_links"]
    headergrp.attrs["Flag_progenitor_links"] = descripdata["Flag_progenitor_links"]
    headergrp.attrs["Flag_forest_ids"] = descripdata["Flag_forest_ids"]

    # overall halo finder and tree builder description
    findergrp = headergrp.create_group("HaloFinder")
    findergrp.attrs["Name"] = descripdata["HaloFinder"]
    findergrp.attrs["Version"] = descripdata["HaloFinder_version"]
    findergrp.attrs["Particle_num_threshold"] = descripdata["Particle_num_threshold"]

    treebuildergrp = headergrp.create_group("TreeBuilder")
    treebuildergrp.attrs["Name"] = descripdata["TreeBuilder"]
    for field in rawtreedata["Header"].keys():
        treebuildergrp.attrs[field] = rawtreedata["Header"][field]


    # simulation params
    simgrp = headergrp.create_group("Simulation")
    for key in simdata.keys():
        simgrp.attrs[key] = simdata[key]
    # unit params
    unitgrp = headergrp.create_group("Units")
    for key in unitdata.keys():
        unitgrp.attrs[key] = unitdata[key]
    # particle types
    partgrp = headergrp.create_group("Parttypes")
    partgrp.attrs["Flag_gas"] = descripdata["Flag_gas"]
    partgrp.attrs["Flag_star"] = descripdata["Flag_star"]
    partgrp.attrs["Flag_bh"] = descripdata["Flag_bh"]
    partmassgrp = headergrp.create_group("Particle_mass")
    for key in partdata['Particle_mass'].keys():
        partmassgrp.attrs[key] = partdata['Particle_mass'][key]

    for i in range(snapshotoffset,snapshotoffset+numsnaps):

        if (ireversesnaporder == True):
            snapnum=(numsnaps-1-i)
        else :
            snapnum=i
        snapgrp = hdffile.create_group("Snap_%03d" % snapnum)
        snapgrp.attrs["Snapnum"] = snapnum
        snapgrp.attrs["NHalos"] = numhalos[i]
        snapgrp.attrs["scalefactor"] = atime[i]
        print("writing snapshot ",snapnum)
        for key in halodata[i].keys():
            halogrp=snapgrp.create_dataset(
                key, data=halodata[i][key], compression="gzip", compression_opts=6)
            if key in treekeys:
                snapgrp[treealiasnames[key]]=halogrp
    hdffile.close()

def WriteForest(basename, numsnaps,
    numhalos, halodata, forestdata, atime,
    descripdata={'Title': 'Halo Forest',
        'HaloFinder' : {
            'Name': 'VELOCIraptor',
            'Version': 1.15,
            'Particle_num_threshold': 20,
            'Subhalo_Particle_num_threshold': 20,
        },
        'TreeBuilder': {
            'Name': 'TreeFrog',
            'Version': 1.1,
            'Temporal_linking_length': 1,
            'Temporal_ID': 1000000000000,
            'Temporally_Unique_Halo_ID_Description': 'Snap_num*Temporal_linking_length+Index+1'
        },
        'ParticleInfo':{
            'Flag_DM': True,
            'Flag_gas': False,
            'Flag_star': False,
            'Flag_bh': False,
            'Flag_zoom': False,
            'Particle_mass': {'dm':-1, 'gas':-1, 'star':-1, 'bh':-1, 'lowres': -1}
        },
    },
    simdata={'Omega_m': 1.0, 'Omega_b': 0., 'Omega_Lambda':0., 'Hubble_param': 1.0, 'BoxSize': 1.0, 'Sigma8': 1.0},
    unitdata={'UnitLength_in_Mpc': 1.0, 'UnitVelocity_in_kms': 1.0, 'UnitMass_in_Msol': 1.0, 'Flag_physical_comoving': True, 'Flag_hubble_flow': False},
    hfconfiginfo=dict(),
    iverbose = 0, iorderhalosbyforest = False, isplit = False, isplitbyforest = False, numsplitsperdim = 1,
    icompress = False,
    ):

    """

    produces a HDF5 file containing the full catalog plus information
    to walk the tree stored in the halo data and forest information.
    Can write seperate files per forest using mpi4py or write a single file
    Assumes the existance of certain fields

    """
    #Get the snapshot offset if present in the header information
    snapshotoffset = 0
    #if("HaloID_snapshot_offset" in rawtreedata["Header"]):
    #    snapshotoffset = rawtreedata["Header"]["HaloID_snapshot_offset"]

    # check to see in tree data already present in halo catalog
    treekeys = ["RootHead", "RootHeadSnap",
                "Head", "HeadSnap",
                "RootTail", "RootTailSnap",
                "Tail", "TailSnap"
                ]
    # alias names of the tree
    treealiaskeys= ['FinalDescendant', 'FinalDescendantSnap',
                    'Descendant', 'DescendantSnap',
                    'FirstProgenitor', 'FirstProgenitorSnap',
                    'Progenitor', 'ProgenitorSnap'
                    ]

    treealiasnames=dict(zip(treekeys,treealiaskeys))
    treebuilderstatuskeys = ['Temporal_linking_length', 'Temporal_halo_id_value']

    #if reordering halo's by forest, run argsort on forest IDs
    #if (ireorderhalosbyforest):
    #    forestordering = np.argsort(forestdata['ForestIDs'])


    nfiles = 1
    forestfile = np.zeros(forestdata['Number_of_forests'], dtype = np.int32)
    forestlist = forestdata['ForestIDs']
    if (isplit and forestdata['Number_of_forests'] > 100):
        if (isplitbyforest):
            if (iverbose >=1):
                print('Splitting by forest ...', flush=True)
            #split by forest
            index = np.argsort(forestdata['ForestSizes'])[::-1]
            forestlist = forestdata['ForestIDs'][index]
            quantindex = int(forestdata['Number_of_forests']*0.02)
            splitsize = forestlist[quantindex]*2
            cumsize = np.cumsum(forestdata['ForestSizes'][index])
            forestfile = np.array(np.floor(cumsize / splitsize), dtype=np.int32)
            nfiles = np.max(forestfile)
            lastfile = np.where(forestfile == nfiles)[0]
            lastfilesum = np.sum(cumsize[lastfile])
            if (lastfilesum>0.5*splitsize):
                nfiles += 1
            else:
                forestfile[lastfile] -= 1
            if (iverbose >=1):
                print('into', nfiles, 'files', flush=True)
        else:
            #split by spatial volume, still not implemented. Just write one file
            nfiles = numsplitsperdim*numsplitsperdim*numsplitsperdim
            nfiles = 1
            if (iverbose >=1):
                print('Splitting spatially ...', flush=True)
            #look at all z=0 halos and split such that do octo split till
            #split the volume into cells of a certain size
            delta = simdata['BoxSize']*atime[-1]/float(numsplitsperdim)
            ix = np.array(halodata[-1]['Xc']/delta, dtype=np.int32)
            iy = np.array(halodata[-1]['Yc']/delta, dtype=np.int32)
            iz = np.array(halodata[-1]['Zc']/delta, dtype=np.int32)
            iindex = ix*numsplitsperdim*numsplitsperdim + iy*numsplitsperdim + iz
            if (iverbose >=1):
                print('into', nfiles, 'files', flush=True)

    #write file containing forest statistics
    fname = basename+'.foreststats.hdf5'
    print('Writing forest statistics to ', fname, flush=True)
    totnumhalos = sum(numhalos)
    totactivehalos = totnumhalos
    hdffile = h5py.File(fname, 'w')
    headergrp = hdffile.create_group('Header')
    headergrp.attrs['HaloCatalogBaseFileName'] = basename
    headergrp.attrs['FilesSplitByForest'] = isplitbyforest
    if (isplit and nfiles > 1):
        if (isplitbyforest):
            headergrp.attrs['FileSplittingCriteria'] = 'SplitByForest'
        else:
            headergrp.attrs['FileSplittingCriteria'] = 'SplitSpatially'
    else:
        headergrp.attrs['FileSplittingCriteria'] = 'NoSplitting'
    headergrp.attrs['NFiles'] = nfiles
    headergrp.attrs["NSnaps"] = numsnaps
    forestgrp = hdffile.create_group('ForestInfo')
    forestgrp.attrs['NForests'] = forestdata['Number_of_forests']
    forestgrp.attrs['MaxForestSize'] = forestdata['Max_forest_size']
    forestgrp.attrs['MaxForestID'] = forestdata['Max_forest_ID']
    forestgrp.attrs['MaxForestFOFGroupSize'] = forestdata['Max_forest_fof_groups_size']
    forestgrp.attrs['MaxForestFOFGroupID'] = forestdata['Max_forest_fof_groups_ID']

    HDF5WriteDataset(forestgrp, 'ForestIDs', forestdata['ForestIDs'], icompress)
    HDF5WriteDataset(forestgrp, 'ForestSizes', forestdata['ForestSizes'], icompress)
    # forestgrp.create_dataset(
    #     'ForestIDs', data=forestdata['ForestIDs'], compression="gzip", compression_opts=6)
    # forestgrp.create_dataset(
    #     'ForestSizes', data=forestdata['ForestSizes'], compression="gzip", compression_opts=6)

    forestsnapgrp = forestgrp.create_group('Snaps')
    for i in range(snapshotoffset,snapshotoffset+numsnaps):
        snapnum = i
        snapkey = "Snap_%03d" % snapnum
        snapgrp = forestsnapgrp.create_group(snapkey)
        snapgrp.attrs['NumActiveForest'] = forestdata['Snapshots']['Number_of_active_forests'][snapkey]
        snapgrp.attrs['MaxForestSize'] = forestdata['Snapshots']['Max_forest_size'][snapkey]
        snapgrp.attrs['MaxForestID'] = forestdata['Snapshots']['Max_forest_ID'][snapkey]
        snapgrp.attrs['MaxForestFOFGroupSize'] = forestdata['Snapshots']['Max_forest_fof_groups_size'][snapkey]
        snapgrp.attrs['MaxForestFOFGroupID'] = forestdata['Snapshots']['Max_forest_fof_groups_ID'][snapkey]
        HDF5WriteDataset(snapgrp, 'NumHalosInForest',
            forestdata['Snapshots']['Number_of_halos_in_forests'][snapkey], icompress)
        HDF5WriteDataset(snapgrp, 'NumFOFGroupsInForest',
            forestdata['Snapshots']['Number_of_fof_groups_in_forests'][snapkey], icompress)
        # snapgrp.create_dataset(
        #     'NumHalosInForest', data=forestdata['Snapshots']['Number_of_halos_in_forests'][snapkey], compression="gzip", compression_opts=6)
        # snapgrp.create_dataset(
        #     'NumFOFGroupsInForest', data=forestdata['Snapshots']['Number_of_fof_groups_in_forests'][snapkey], compression="gzip", compression_opts=6)

    foo, counts = np.unique(forestfile, return_counts=True)
    HDF5WriteDataset(forestgrp, 'NForestsPerFile', counts, icompress)
    # forestgrp.create_dataset(
    #     'NForestsPerFile', data=counts, compression="gzip", compression_opts=6)
    hdffile.close()
    print('Done', flush=True)

    print('Writing halo+tree+forest data ... ', flush=True)
    for ifile in range(nfiles):
        fname = basename+'.hdf5.%d'%ifile
        print('Write to ', fname, flush=True)
        hdffile = h5py.File(fname, 'w')
        headergrp = hdffile.create_group("Header")

        # store useful information such as number of snapshots, halos,
        # cosmology (Omega_m, Omega_b, Hubble_param, Omega_Lambda, Box size)
        # units (Physical [1/0] for physical/comoving flag, length in Mpc, km/s, solar masses, Gravity
        # and TEMPORALHALOIDVAL used to traverse tree information (converting halo ids to haloindex or snapshot), Reverse_order [1/0] for last snap listed first)
        # set the attributes of the header
        headergrp.attrs["ThisFile"] = ifile
        headergrp.attrs["NFiles"] = nfiles
        headergrp.attrs["NSnaps"] = numsnaps
        headergrp.attrs["Flag_subhalo_links"] = True
        headergrp.attrs["Flag_progenitor_links"] = True
        headergrp.attrs["Flag_forest_ids"] = True

        # overall halo finder and tree builder description
        findergrp = headergrp.create_group("HaloFinder")
        findergrp.attrs["Name"] = descripdata["HaloFinder"]['Name']
        findergrp.attrs["Version"] = descripdata["HaloFinder"]['Version']
        findergrp.attrs["Particle_num_threshold"] = descripdata["HaloFinder"]["Particle_num_threshold"]

        treebuildergrp = headergrp.create_group("TreeBuilder")
        treebuildergrp.attrs["Name"] = descripdata["TreeBuilder"]['Name']
        treebuildergrp.attrs["Version"] = descripdata["TreeBuilder"]['Version']
        for field in treebuilderstatuskeys:
            treebuildergrp.attrs[field] = descripdata["TreeBuilder"][field]

        # simulation params
        simgrp = headergrp.create_group("Simulation")
        for key in simdata.keys():
            simgrp.attrs[key] = simdata[key]
        # unit params
        unitgrp = headergrp.create_group("Units")
        for key in unitdata.keys():
            unitgrp.attrs[key] = unitdata[key]
        # particle types
        partgrp = headergrp.create_group("Parttypes")
        for key in descripdata['ParticleInfo'].keys():
            if (type(descripdata['ParticleInfo'][key]) is dict): continue
            partgrp.attrs[key] = descripdata['ParticleInfo'][key]
        partmassgrp = headergrp.create_group("Particle_mass")
        for key in descripdata['ParticleInfo']['Particle_mass'].keys():
            partmassgrp.attrs[key] = descripdata['ParticleInfo']['Particle_mass'][key]
        forestgrp = hdffile.create_group("ForestInfoInFile")
        if (nfiles > 1):
            wdata = np.where(forestfile == ifile)
            activeforest = forestlist[wdata]
            activeforestsizes = forestdata['ForestSizes'][wdata]
            activeforestfofsizes = forestdata['Max_forest_fof_groups_size'][wdata]
        else:
            activeforest = forestlist
            activeforestsizes = forestdata['ForestSizes']
            activeforestfofsizes = forestdata['Max_forest_fof_groups_size']
        HDF5WriteDataset(forestgrp, 'ForestIDsInFile', activeforest, icompress)
        HDF5WriteDataset(forestgrp, 'ForestSizesInFile', activeforestsizes, icompress)
        # forestgrp.create_dataset('ForestIDsInFile', data=activeforest,
        #     compression="gzip", compression_opts=6)
        # forestgrp.create_dataset('ForestSizesInFile', data=activeforestsizes,
        #     compression="gzip", compression_opts=6)
        forestgrp.attrs['NForestsInFile'] = activeforest.size
        forestgrp.attrs['MaxForestSizeInFile'] = np.max(activeforestsizes)
        forestgrp.attrs['MaxForestIDInFile'] = activeforest[np.argmax(activeforestsizes)]
        forestgrp.attrs['MaxForestFOFGroupSizeInFile'] = np.max(activeforestfofsizes)
        forestgrp.attrs['MaxForestFOFGroupIDInFile'] = activeforest[np.argmax(activeforestfofsizes)]

        for i in range(snapshotoffset,snapshotoffset+numsnaps):
            snapnum=i
            if (iverbose >=1):
                print("writing snapshot ",snapnum, flush=True)
            snapgrp = hdffile.create_group("Snap_%03d" % snapnum)
            snapgrp.attrs["Snapnum"] = snapnum
            snapgrp.attrs["scalefactor"] = atime[i]
            if (nfiles > 1):
                activehalos = np.where(np.isin(halodata[i]['ForestID'], activeforest))[0]
                uniquevalues, uniquecounts = np.unique(halodata[i]['ForestID'][activehalos],
                    return_counts = True)
                wdata = np.where(np.isin(activeforest, uniquevalues))[0]
                activeforestinsnapsizes = np.zeros(activeforest.size, dtype=np.uint64)
                activeforestinsnapsizes[wdata] = uniquecounts
                nactive = activehalos.size
            else:
                uniquevalues, uniquecounts = np.unique(halodata[i]['ForestID'],
                    return_counts = True)
                wdata = np.where(np.isin(activeforest, uniquevalues))[0]
                activeforestinsnapsizes = np.zeros(activeforest.size, dtype=np.uint64)
                activeforestinsnapsizes[wdata] = uniquecounts
                nactive = numhalos[i]
            snapgrp.attrs["NHalos"] = nactive
            HDF5WriteDataset(snapgrp, "NHalosPerForestInSnap", activeforestinsnapsizes, icompress)
            # snapgrp.create_dataset("NHalosPerForestInSnap", data = activeforestinsnapsizes,
            #     compression="gzip", compression_opts=6)

            #write halo properties
            for key in halodata[i].keys():
                if (key == 'ConfigurationInfo' or key == 'SimulationInfo' or key == 'UnitInfo'): continue
                datablock = None
                if (nfiles == 1):
                    datablock = halodata[i][key]
                else:
                    datablock = halodata[i][key][activehalos]
                halogrp = HDF5WriteDataset(snapgrp, key, datablock, icompress)
                # halogrp=snapgrp.create_dataset(
                #     key, data=datablock, compression="gzip", compression_opts=6)
                if key in treekeys:
                    snapgrp[treealiasnames[key]] = halogrp
        hdffile.close()
        print('Done', flush=True)

def ReadForest(basename, desiredfields=[], iverbose=False):
    """
    Read forest file HDF file with base filename fname.

    Parameters
    ----------

    Returns
    -------
    halo catalogs, trees, etc
    """

    headergrpname = "Header/"
    simgrpname = "Simulation/"
    unitgrpname = "Units/"

    fname = basename+'.hdf5.%d'%0
    hdffile = h5py.File(fname, 'r')
    # first get number of files
    numsnaps = np.int64(hdffile['Header'].attrs["NSnaps"])
    nfiles = np.int64(hdffile['Header'].attrs["NFiles"])

    # allocate memory
    halodata = [dict() for i in range(numsnaps)]
    numhalos = np.zeros(numsnaps, dtype=np.int64)
    atime = np.zeros(numsnaps)
    simdata = dict()
    unitdata = dict()

    # load simulation data
    fieldnames = [str(n)
                  for n in hdffile[headergrpname+simgrpname].attrs.keys()]
    for fieldname in fieldnames:
        simdata[fieldname] = hdffile[headergrpname +
                                       simgrpname].attrs[fieldname]

    # load unit data
    fieldnames = [str(n)
                  for n in hdffile[headergrpname+unitgrpname].attrs.keys()]
    for fieldname in fieldnames:
        unitdata[fieldname] = hdffile[headergrpname +
                                      unitgrpname].attrs[fieldname]
    if (len(desiredfields) > 0):
        fieldnames = desiredfields
    else:
        fieldnames = [str(n) for n in hdffile['Snap_000'].keys()]
    for i in range(numsnaps):
        for fieldname in fieldnames:
            halodata[i][fieldname] = np.array([], dtype=hdffile['Snap_000'][fieldname].dtype)

    hdffile.close()

    for i in range(nfiles):
        fname = basename+'.hdf5.%d'%0
        hdffile = h5py.File(fname, 'r')
        # for each snap load the appropriate group
        start = time.process_time()
        for i in range(numsnaps):
            snapgrpname = "Snap_%03d/" % i
            if (iverbose == True):
                print("Reading ", snapgrpname, flush=True)
            isnap = hdffile[snapgrpname].attrs["Snapnum"]
            atime[isnap] = hdffile[snapgrpname].attrs["scalefactor"]
            numhalos[isnap] += hdffile[snapgrpname].attrs["NHalos"]
            for catvalue in fieldnames:
                halodata[isnap][catvalue] = np.concatenate([halodata[isnap][catvalue],np.array(
                    hdffile[snapgrpname+catvalue])])
        hdffile.close()
    print("read halo data ", time.process_time()-start, flush=True)

    return halodata, numhalos, atime, simdata, unitdata

def ForestSorter(basename, isortorder = 'random', ibackup = True,
    icompress = False):
    """
    Sorts a forest file and remaps halo IDs.
    The sort fields (or sort keys) we ordered such that the first key will peform the
    outer-most sort and the last key will perform the inner-most sort.

    Parameters
    ----------
    basename : String
        Base file name of the forest file.
        Open HDF5 file of the forest, reading meta information and
        assumed the HDF5_File to have the following data structure
        HDF5_file -> Snapshot_Keys -> Halo properties.
        The file is updated with new IDs and stores old IDs as IDs_old
        plus saves meta information mapping IDs
    ibackup : bool
        Whether to back up the file before updating it.
    Returns
    ----------
    void:

    ----------
        sort_fields = ["ForestID", "Mass_200mean"]
        ForestID = [1, 4, 39, 1, 1, 4]
        Mass_200mean = [4e9, 10e10, 8e8, 7e9, 3e11, 5e6]
        Then the indices would be [0, 3, 4, 5, 1, 2]
    """

    # data fields that will need values updated as ids will be mapped.
    # as some fields are aliased, don't update them
    temporalkeys = [
        #'RootHead',
        #'Head',
        #'RootTail',
        #'Tail',
        'FinalDescendant',
        'Descendant',
        'FirstProgenitor',
        'Progenitor',
        #'LeftTail',
        #'RightTail',
        'PreviousProgenitor',
        'NextProgenitor',
    ]
    subhalokeys = [
        'hostHaloID',
        'NextSubhalo',
        'PreviousSubhalo',
        ]
    sortorderkeys = ['ids', 'sizes', 'random']
    if (isortorder not in sortorderkeys):
        print('Error: ',isortorder, 'not valid. Sort order can be ',sortorderkeys, flush=True)
        print('Exiting without sorting', flush=True)

    # fields used to determine ordering of halos in file
    sort_fields = ['ForestID', 'hostHaloID', 'npart']

    #open old files to get necessary information
    fname = basename+'.foreststats.hdf5'
    hdffile = h5py.File(fname, 'r')
    forestids = np.array(hdffile['ForestInfo']['ForestIDs'])
    forestsizes = np.array(hdffile['ForestInfo']['ForestSizes'])

    if (isortorder == 'ids'):
        forestordering = np.argsort(forestsizes)
    elif (isortorder == 'sizes'):
        forestordering = np.argsort(forestsizes)
    elif (isortorder == 'random'):
        forestordering = np.random.choice(np.argsort(forestsizes),
            forestids.size, replace=False)

    numsnaps = np.int64(hdffile['Header'].attrs["NSnaps"])
    nfiles = np.int64(hdffile['Header'].attrs["NFiles"])
    hdffile.close()

    fname = basename+'.hdf5.%d'%0
    hdffile = h5py.File(fname, 'r')
    TEMPORALHALOIDVAL = np.int64(hdffile['Header/TreeBuilder'].attrs['Temporal_halo_id_value'])
    snapkey = "Snap_%03d" % (numsnaps-1)
    allpropkeys = list(hdffile[snapkey].keys())
    idkeylist = []
    propkeys = []
    aliasedkeys = []
    for propkey in allpropkeys:
        if (hdffile[snapkey][propkey].id not in idkeylist):
            idkeylist.append(hdffile[snapkey][propkey].id)
            propkeys.append(propkey)
        else:
            aliasedkeys.append(propkey)
    hdffile.close()

    # back up files if necessary
    if (ibackup):
        print('Backing up original data', flush=True)
        fname = basename+'.foreststats.hdf5'
        newfname = fname+'.backup'
        subprocess.call(['cp', fname, newfname])
        for ifile in range(nfiles):
            fname = basename+'.hdf5.%d'%ifile
            newfname = fname+'.backup'
            subprocess.call(['cp', fname, newfname])

    # reorder file containing meta information
    print('Reordering forest stats data ...', flush=True)
    time1 = time.process_time()
    fname = basename+'.foreststats.hdf5'
    hdffile = h5py.File(fname, 'r+')
    forestgrp = hdffile['ForestInfo']
    data = forestgrp['ForestIDs']
    forestids = forestids[forestordering]
    data[:] = forestids
    data = forestgrp['ForestSizes']
    data[:] = forestsizes[forestordering]
    snapskeys = list(forestgrp['Snaps'].keys())
    for snapkey in snapskeys:
        snapgrp = forestgrp['Snaps'][snapkey]
        numhalos = np.array(snapgrp['NumHalosInForest'])[forestordering]
        numfofs = np.array(snapgrp['NumFOFGroupsInForest'])[forestordering]
        data = snapgrp['NumHalosInForest']
        data[:] = numhalos
        data = snapgrp['NumFOFGroupsInForest']
        data[:] = numfofs
    hdffile.close()
    print('Done', time.process_time()-time1, flush=True)

    for ifile in range(nfiles):
        fname = basename+'.hdf5.%d'%ifile
        hdffile = h5py.File(fname, 'a')
        print('First pass building id map for file', fname, flush=True)

        #first pass to resort arrays
        #store the ids and the newids to map stuff
        alloldids = np.array([], dtype=np.int64)
        allnewids = np.array([], dtype=np.int64)
        time1 = time.process_time()
        for i in range(numsnaps):
            snapkey = "Snap_%03d" % i
            numhalos = np.int64(hdffile[snapkey].attrs['NHalos'])
            if (numhalos == 0): continue
            ids = np.array(hdffile[snapkey]['ID'], dtype=np.int64)
            sort_data = np.zeros([len(sort_fields),ids.size], dtype=np.int64)
            sort_data[0] = -np.array(hdffile[snapkey]['npart'], dtype=np.int64)
            sort_data[1] = np.array(hdffile[snapkey]['hostHaloID'], dtype=np.int64)
            activeforestids = np.array(hdffile[snapkey]['ForestID'], dtype=np.int64)
            xy, x_ind, y_ind = np.intersect1d(activeforestids, forestids, return_indices=True)
            unique, inverse = np.unique(activeforestids, return_inverse=True)
            sort_data[2] = y_ind[inverse]
            indices = np.array(np.lexsort(sort_data))
            newids = i*TEMPORALHALOIDVAL+np.arange(numhalos, dtype=np.int64)+1

            alloldids = np.concatenate([alloldids,np.array(ids[indices], dtype=np.int64)])
            allnewids = np.concatenate([allnewids,newids])

            for propkey in propkeys:
                if (propkey == 'NHalosPerForestInSnap'): continue
                if (propkey == 'ID'): continue
                if (propkey in aliasedkeys): continue
                newdata = np.array(hdffile[snapkey][propkey])[indices]
                data = hdffile[snapkey][propkey]
                data[:] = newdata
            HDF5WriteDataset(hdffile[snapkey], 'ID_old', ids[indices], icompress)
            # hdffile[snapkey].create_dataset('ID_old',
            #     data=ids[indices], dtype=np.int64, compression='gzip', compression_opts=6)
            data = hdffile[snapkey]['ID']
            data[:] = newids

        #now go over temporal and subhalo fields and update as necessary
        print('Finished pass and now have map of new ids to old ids', time.process_time()-time1, flush=True)
        time1 = time.process_time()
        for i in range(numsnaps):
            snapkey = "Snap_%03d" % i
            numhalos = np.int32(hdffile[snapkey].attrs['NHalos'])
            if (numhalos == 0): continue
            print('Processing',snapkey, flush=True)
            time2 = time.process_time()
            for propkey in temporalkeys:
                olddata = np.array(hdffile[snapkey][propkey])
                olddata_unique, olddata_unique_inverse = np.unique(olddata, return_inverse = True)
                xy, x_ind, y_ind = np.intersect1d(alloldids, olddata_unique, return_indices=True)
                newdata = allnewids[x_ind[olddata_unique_inverse]]
                data = hdffile[snapkey][propkey]
                data[:] = newdata
            for propkey in subhalokeys:
                olddata = np.array(hdffile[snapkey][propkey])
                if (propkey == 'hostHaloID'):
                    newdata = -np.ones(numhalos, dtype=np.int64)
                    wdata = np.where(olddata !=-1)[0]
                    if (wdata.size >0):
                        olddata_unique, olddata_unique_inverse = np.unique(olddata[wdata], return_inverse = True)
                        xy, x_ind, y_ind = np.intersect1d(alloldids, olddata_unique, return_indices=True)
                        newdata[wdata] = allnewids[x_ind[olddata_unique_inverse]]
                else:
                    olddata = np.array(hdffile[snapkey][propkey])
                    olddata_unique, olddata_unique_inverse = np.unique(olddata, return_inverse = True)
                    xy, x_ind, y_ind = np.intersect1d(alloldids, olddata_unique, return_indices=True)
                    newdata = allnewids[x_ind[olddata_unique_inverse]]
                data = hdffile[snapkey][propkey]
                data[:] = newdata
            print('Done', snapkey, 'containing', numhalos, 'in', time.process_time()-time2, flush=True)

        #now update the forest info in the file
        forestgrp = hdffile['ForestInfoInFile']
        data = forestgrp['ForestIDsInFile']
        activeforestids = np.array(data)
        xy, x_ind, y_ind = np.intersect1d(activeforestids, forestids, return_indices=True)
        ordering = x_ind[np.argsort(y_ind)]
        data[:] = activeforestids[ordering]
        data = forestgrp['ForestSizesInFile']
        data[:] = np.array(data)[ordering]
        for i in range(numsnaps):
            snapkey = "Snap_%03d" % i
            data = hdffile[snapkey]['NHalosPerForestInSnap']
            newdata = np.array(data)[ordering]
            data[:] = newdata
        # for i in range(numsnaps):
        #     snapkey = "Snap_%03d" % i
        #     snapgrp = forestgrp[snapkey]
        #     numhalos = np.array(snapgrp['NumHalosInForest'])[ordering]
        #     numfofs = np.array(snapgrp['NumFOFGroupsInForest'])[ordering]
        #     data = snapgrp['NumHalosInForest']
        #     data[:] = numhalos
        #     data = snapgrp['NumFOFGroupsInForest']
        #     data[:] = numfofs

        hdffile.create_group('ID_mapping')
        HDF5WriteDataset(hdffile['ID_mapping'], 'IDs_old', alloldids, icompress)
        HDF5WriteDataset(hdffile['ID_mapping'], 'IDs_new', allnewids, icompress)
        # hdffile['ID_mapping'].create_dataset('IDs_old', data = alloldids)
        # hdffile['ID_mapping'].create_dataset('IDs_new', data = allnewids)
        print('Finished updating data ', time.process_time()-time1, flush=True)
        hdffile.close()

def ForestFileAddMetaData(basename, icompress = False):
    """
    Add some metadata to forest files

    Parameters
    ----------
    basename : String
        Base file name of the forest file.
        Open HDF5 file of the forest, reading meta information and
        assumed the HDF5_File to have the following data structure
        HDF5_file -> Snapshot_Keys -> Halo properties.
        The file is updated to force bi-directional tree
    Returns
    ----------
    void:

    """

    #open old files to get necessary information
    fname = basename+'.foreststats.hdf5'
    hdffile = h5py.File(fname, 'r')
    numsnaps = np.int64(hdffile['Header'].attrs["NSnaps"])
    nfiles = np.int64(hdffile['Header'].attrs["NFiles"])
    hdffile.close()

    fname = basename+'.hdf5.%d'%0
    hdffile = h5py.File(fname, 'r')
    TEMPORALHALOIDVAL = np.int64(hdffile['Header/TreeBuilder'].attrs['Temporal_halo_id_value'])
    hdffile.close()

    for ifile in range(nfiles):
        fname = basename+'.hdf5.%d'%ifile
        hdffile = h5py.File(fname, 'a')
        time1 = time.process_time()
        nforests = np.uint64(hdffile['ForestInfoInFile'].attrs['NForestsInFile'])
        for i in range(numsnaps):
            snapkey = "Snap_%03d" % i
            numhalos = np.uint64(hdffile[snapkey].attrs['NHalos'])
            if ('ForestOffsetPerSnap' in list(hdffile[snapkey].keys())):
                del hdffile[snapkey]['ForestOffsetPerSnap']
            offset = np.zeros(nforests, dtype=np.int64)
            if (numhalos > 0):
                offset[1:] = np.cumsum(np.array(hdffile[snapkey]['NHalosPerForestInSnap']))[:-1]
            HDF5WriteDataset(hdffile[snapkey], "ForestOffsetPerSnap", offset, icompress)
            # hdffile[snapkey].create_dataset("ForestOffsetPerSnap", data = offset)

        if ("ForestSizesAllSnaps" not in list(hdffile['ForestInfoInFile'].keys()) and
            "ForestOffsetsAllSnaps" not in list(hdffile['ForestInfoInFile'].keys())):
            forestsizesinfile = np.zeros([numsnaps,nforests], dtype=np.int64)
            forestoffsetsinfile = np.zeros([numsnaps,nforests], dtype=np.int64)
            for i in range(numsnaps):
                snapkey = "Snap_%03d" % i
                forestsizesinfile[i] = np.array(hdffile[snapkey]['NHalosPerForestInSnap'])
                forestoffsetsinfile[i][1:] = np.cumsum(forestsizesinfile[i])[:-1]
            forestsizesinfile = forestsizesinfile.transpose()
            forestoffsetsinfile = forestoffsetsinfile.transpose()
            HDF5WriteDataset(hdffile['ForestInfoInFile'], "ForestSizesAllSnaps",
                forestsizesinfile, icompress)
            HDF5WriteDataset(hdffile['ForestInfoInFile'], "ForestOffsetsAllSnaps",
                forestoffsetsinfile, icompress)
            # hdffile['ForestInfoInFile'].create_dataset("ForestSizesAllSnaps", data = forestsizesinfile)
            # hdffile['ForestInfoInFile'].create_dataset("ForestOffsetsAllSnaps", data = forestoffsetsinfile)

        for i in range(numsnaps):
            snapkey = "Snap_%03d" % i
            hdffile['ForestInfoInFile'].create_group(snapkey)
            hdffile['ForestInfoInFile'][snapkey]['NHalosPerForestInSnap'] = hdffile[snapkey]['NHalosPerForestInSnap']
            hdffile['ForestInfoInFile'][snapkey]["ForestOffsetPerSnap"] = hdffile[snapkey]['ForestOffsetPerSnap']
        print('Finished updating data ', time.process_time()-time1, flush=True)
        hdffile.close()

def ForestFileAddHaloData(basename: str,
    halocatalognames : list,
    desiredfields : list = [],
    icompress : bool = False):
    """
    Add some metadata to forest files

    Parameters
    ----------
    basename : String
        Base file name of the forest file.
    halocatalognames : list
        Base file name of the halo catalogs from which data will be added
    desiredfields : list of strings
        field names to be extracted from the halo catalogs and added to the forest
        file. This can be a slow process but is memory efficient

        Open HDF5 file of the forest and adds halo properites.
    Returns
    ----------
    void:

    """

    #open old files to get necessary information
    fname = basename+'.foreststats.hdf5'
    hdffile = h5py.File(fname, 'r')
    numsnaps = np.int64(hdffile['Header'].attrs["NSnaps"])
    nfiles = np.int64(hdffile['Header'].attrs["NFiles"])
    hdffile.close()

    fname = basename+'.hdf5.%d'%0
    hdffile = h5py.File(fname, 'r')
    TEMPORALHALOIDVAL = np.int64(hdffile['Header/TreeBuilder'].attrs['Temporal_halo_id_value'])
    hdffile.close()

    hdffiles = [None for ifile in range(nfiles)]
    for ifile in range(nfiles):
        fname = basename+'.hdf5.%d'%ifile
        hdffiles[ifile] = h5py.File(fname, 'a')
    for i in range(numsnaps):
        snapkey = "Snap_%03d" % i
        haloname = halocatalognames[i]
        halodata, numhalos = ReadPropertyFile(haloname, 2, 0, 0, desiredfields)
        print('Adding halo data from', haloname, 'containing', numhalos, flush=True)
        time1 = time.process_time()
        for ifile in range(nfiles):
            hdffile = hdffiles[ifile]
            time2 = time.process_time()
            numhalosinforestfile = np.uint64(hdffile[snapkey].attrs['NHalos'])
            if (numhalos != numhalosinforestfile) :
                activehalodata = dict()
                activeindex = np.int64(hdffile[snapkey]['IDs'] % TEMPORALHALOIDVAL - 1)
                for key in desiredfields:
                    activehalodata[key] = halodata[key][activeindex]
            else:
                activehalodata = halodata
            for key in desiredfields:
                halogrp = HDF5WriteDataset(hdffile[snapkey], key, activehalodata[key], icompress)
            halodata[key] = None
        print('Finished adding data to files in ', time.process_time() - time1, flush=True)
    for ifile in range(nfiles):
        hdffiles[ifile].close()

def ForceBiDirectionalTreeInForestFile(basename, icompress = False):
    """
    Forces bi-directional tree in file

    Parameters
    ----------
    basename : String
        Base file name of the forest file.
        Open HDF5 file of the forest, reading meta information and
        assumed the HDF5_File to have the following data structure
        HDF5_file -> Snapshot_Keys -> Halo properties.
        The file is updated to force bi-directional tree
    Returns
    ----------
    void:

    """

    #open old files to get necessary information
    fname = basename+'.foreststats.hdf5'
    hdffile = h5py.File(fname, 'r')
    numsnaps = np.int64(hdffile['Header'].attrs["NSnaps"])
    nfiles = np.int64(hdffile['Header'].attrs["NFiles"])
    hdffile.close()

    fname = basename+'.hdf5.%d'%0
    hdffile = h5py.File(fname, 'r')
    TEMPORALHALOIDVAL = np.int64(hdffile['Header/TreeBuilder'].attrs['Temporal_halo_id_value'])
    hdffile.close()

    for ifile in range(nfiles):
        fname = basename+'.hdf5.%d'%ifile
        hdffile = h5py.File(fname, 'a')
        time1 = time.process_time()
        for i in range(numsnaps):
            snapkey = "Snap_%03d" % i
            numhalos = np.int32(hdffile[snapkey].attrs['NHalos'])
            if (numhalos == 0): continue
            ids = np.array(hdffile[snapkey]['ID'])
            descens = np.array(hdffile[snapkey]['Descendant'])
            descensnaps = np.int32(descens / TEMPORALHALOIDVAL)
            descenindex = np.int64(descens % TEMPORALHALOIDVAL - 1)
            #descenprogen = dict(zip(descens,ids))
            maxsnaps = np.max(descensnaps)
            for isnap in range(i+1,maxsnaps+1):
                snapkey2 = "Snap_%03d" % isnap
                activedescens = np.where(descensnaps == isnap)[0]
                if (activedescens.size == 0): continue
                ids2 = np.array(hdffile[snapkey2]['ID'])[descenindex[activedescens]]
                progens = np.array(hdffile[snapkey2]['Progenitor'])[descenindex[activedescens]]
                wdata = np.where(progens == ids2)[0]
                if (wdata.size == 0): continue
                newdata = np.array(hdffile[snapkey2]['Progenitor'])
                newdata[descenindex[activedescens][wdata]] = ids[activedescens][wdata]
                data = hdffile[snapkey2]['Progenitor']
                data[:] = newdata
        print('Finished updating data ', time.process_time()-time1, flush=True)
        hdffile.close()

def PruneForest(basename, forestsizelimit = 2, ibackup = True,
    icompress = False):
    """
    Prunes a Forest, removing all those containing <= forestsizelimit halos

    Parameters
    ----------
    basename : String
        Base file name of the forest file.
        Open HDF5 file of the forest, reading meta information and
        assumed the HDF5_File to have the following data structure
        HDF5_file -> Snapshot_Keys -> Halo properties.
        The file is updated to force bi-directional tree
    forestsizelimit : int
        Limit used to remove all forests below or at this limit
    ibackup : bool
        Whether to back up the file before updating it.
    Returns
    ----------
    void:

    """

    #open old files to get necessary information
    fname = basename+'.foreststats.hdf5'
    hdffile = h5py.File(fname, 'r')
    forestids = np.array(hdffile['ForestInfo']['ForestIDs'])
    forestsizes = np.array(hdffile['ForestInfo']['ForestSizes'])
    forestselection = np.where(forestsizes>=forestsizelimit)[0]
    numsnaps = np.int64(hdffile['Header'].attrs["NSnaps"])
    nfiles = np.int64(hdffile['Header'].attrs["NFiles"])
    hdffile.close()
    # if all forests are of the correct size
    # then do nothing
    if (forestselection.size == forestsizes.size):
        print('Forest file has no forests to be pruned based on min size of', forestsizelimit, flush=True)
        print('Doing nothing', flush=True)
        return
    elif (forestselection.size == 0):
        print('Error, all forests to be pruned based on minimum size of', forestsizelimit, flush=True)
        print('Doing nothing', flush=True)
        return
    else :
        print('Pruning forest file of forests smaller than', forestsizelimit, flush=True)
        print('This is removing', forestsizes.size - forestselection.size, flush=True)

    fname = basename+'.hdf5.%d'%0
    hdffile = h5py.File(fname, 'r')
    TEMPORALHALOIDVAL = np.int64(hdffile['Header/TreeBuilder'].attrs['Temporal_halo_id_value'])
    snapkey = "Snap_%03d" % (numsnaps-1)
    allpropkeys = list(hdffile[snapkey].keys())
    idkeylist = []
    propkeys = []
    aliasedkeys = []
    aliasedkeyspropkeymap = dict()
    for propkey in allpropkeys:
        if (hdffile[snapkey][propkey].id not in idkeylist):
            idkeylist.append(hdffile[snapkey][propkey].id)
            propkeys.append(propkey)
        else:
            aliasedkeys.append(propkey)
    for aliaskey in aliasedkeys:
        index = idkeylist.index(hdffile[snapkey][aliaskey].id)
        aliasedkeyspropkeymap[aliaskey] = propkeys[index]
    hdffile.close()

    if (ibackup):
        print('Backing up original data', flush=True)
        fname = basename+'.foreststats.hdf5'
        newfname = fname+'.prepruningbackup'
        subprocess.call(['cp', fname, newfname])
        for ifile in range(nfiles):
            fname = basename+'.hdf5.%d'%ifile
            newfname = fname+'.prepruningbackup'
            subprocess.call(['cp', fname, newfname])

    print('Updating data ...', flush=True)
    time1 = time.process_time()
    fname = basename+'.foreststats.hdf5'
    hdffile = h5py.File(fname, 'a')
    forestgrp = hdffile['ForestInfo']
    forestgrp.attrs['NForests'] = forestselection.size
    #forestgrp.attrs['MaxForestSize'] = forestdata['Max_forest_size']
    #forestgrp.attrs['MaxForestID'] = forestdata['Max_forest_ID']
    #forestgrp.attrs['MaxForestFOFGroupSize'] = forestdata['Max_forest_fof_groups_size']
    #forestgrp.attrs['MaxForestFOFGroupID'] = forestdata['Max_forest_fof_groups_ID']

    del forestgrp['ForestIDs']
    HDF5WriteDataset(forestgrp, 'ForestIDs', forestids[forestselection], icompress)
    # forestgrp.create_dataset('ForestIDs', data=forestids[forestselection])
    forestids = forestids[forestselection]
    del forestgrp['ForestSizes']
    HDF5WriteDataset(forestgrp, 'ForestSizes', forestsizes[forestselection], icompress)
    # forestgrp.create_dataset('ForestSizes', data=forestsizes[forestselection])
    forestsizes = forestsizes[forestselection]
    forestsnapgrp = forestgrp['Snaps']
    for i in range(numsnaps):
        snapnum = i
        snapkey = "Snap_%03d" % snapnum
        snapgrp = forestsnapgrp[snapkey]
        data = snapgrp['NumHalosInForest']
        newdata = np.array(data)[forestselection]
        snapgrp.attrs['NumActiveForest'] = np.where(newdata>0)[0].size
        del snapgrp['NumHalosInForest']
        HDF5WriteDataset(snapgrp, 'NumHalosInForest', newdata)
        # snapgrp.create_dataset('NumHalosInForest', data=newdata)
        data = snapgrp['NumFOFGroupsInForest']
        newdata = np.array(data)[forestselection]
        del snapgrp['NumFOFGroupsInForest']
        HDF5WriteDataset(snapgrp, 'NumFOFGroupsInForest', newdata)
        # snapgrp.create_dataset('NumFOFGroupsInForest', data=newdata)
        #snapgrp.attrs['MaxForestSize'] = forestdata['Snapshots']['Max_forest_size'][snapkey]
        #snapgrp.attrs['MaxForestID'] = forestdata['Snapshots']['Max_forest_ID'][snapkey]
        #snapgrp.attrs['MaxForestFOFGroupSize'] = forestdata['Snapshots']['Max_forest_fof_groups_size'][snapkey]
        #snapgrp.attrs['MaxForestFOFGroupID'] = forestdata['Snapshots']['Max_forest_fof_groups_ID'][snapkey]
    print('metadata updated', time.process_time()-time1, flush=True)
    hdffile.close()

    newnumforestinfiles = np.zeros(nfiles, dtype=np.int64)
    for ifile in range(nfiles):
        time1 = time.process_time()
        fname = basename+'.hdf5.%d'%ifile
        print('updating', fname, flush=True)
        hdffile = h5py.File(fname, 'a')
        forestgrp = hdffile["ForestInfoInFile"]
        data = forestgrp['ForestIDsInFile']
        olddata = np.array(data)
        newdata, x_ind, y_ind = np.intersect1d(forestids, olddata, return_indices=True)
        newnumforestinfiles[ifile] = y_ind.size
        del forestgrp['ForestIDsInFile']
        HDF5WriteDataset(forestgrp, 'ForestIDsInFile', newdata, icompress)
        # forestgrp.create_dataset('ForestIDsInFile', data=newdata)
        data = forestgrp['ForestSizesInFile']
        newdata = np.array(data)[y_ind]
        del forestgrp['ForestSizesInFile']
        HDF5WriteDataset(forestgrp, 'ForestSizesInFile', newdata, icompress)
        # forestgrp.create_dataset('ForestSizesInFile', data=newdata)
        forestgrp.attrs['NForestsInFile'] = newdata.size
        #forestgrp.attrs['MaxForestSizeInFile'] = np.max(activeforestsizes)
        #forestgrp.attrs['MaxForestIDInFile'] = activeforest[np.argmax(activeforestsizes)]
        #forestgrp.attrs['MaxForestFOFGroupSizeInFile'] = np.max(activeforestfofsizes)
        #forestgrp.attrs['MaxForestFOFGroupIDInFile'] = activeforest[np.argmax(activeforestfofsizes)]
        for i in range(numsnaps):
            snapkey = "Snap_%03d" % i
            numhalos = np.int32(hdffile[snapkey].attrs['NHalos'])
            data = hdffile[snapkey]['NHalosPerForestInSnap']
            newdata = np.array(data)[y_ind]
            del hdffile[snapkey]['NHalosPerForestInSnap']
            HDF5WriteDataset(hdffile[snapkey], 'NHalosPerForestInSnap', newdata, icompress)
            # hdffile[snapkey].create_dataset('NHalosPerForestInSnap', data=newdata)
            if (numhalos == 0): continue
            allcurrentforests = np.array(hdffile[snapkey]['ForestID'])
            activeforestindex = np.where(np.isin(allcurrentforests, forestids))[0]
            for propkey in propkeys:
                if (propkey == 'NHalosPerForestInSnap'): continue
                if (propkey in aliasedkeys): continue
                newdata = np.array(hdffile[snapkey][propkey])[activeforestindex]
                del hdffile[snapkey][propkey]
                HDF5WriteDataset(hdffile[snapkey], propkey, newdata, icompress)
                # hdffile[snapkey].create_dataset(propkey, data=newdata)
            for aliaskey in aliasedkeys:
                del hdffile[snapkey][aliaskey]
                hdffile[snapkey][aliaskey] = hdffile[snapkey][aliasedkeyspropkeymap[aliaskey]]
            hdffile[snapkey].attrs['NHalos'] = activeforestindex.size
        print('Finished updating data ', time.process_time()-time1, flush=True)
        hdffile.close()

    #last update on number of forests per file
    fname = basename+'.foreststats.hdf5'
    hdffile = h5py.File(fname, 'a')
    forestgrp = hdffile['ForestInfo']
    data = forestgrp['NForestsPerFile']
    data[:] = newnumforestinfiles
    hdffile.close()


def WriteCombinedUnifiedTreeandHaloCatalog(fname, numsnaps, rawtree, numhalos, halodata, atime,
                                           descripdata={'Title': 'Tree and Halo catalog of sim', 'VELOCIraptor_version': 1.15, 'Tree_version': 1.1,
                                                        'Particle_num_threshold': 20, 'Temporal_linking_length': 1, 'Flag_gas': False, 'Flag_star': False, 'Flag_bh': False},
                                           cosmodata={'Omega_m': 1.0, 'Omega_b': 0., 'Omega_Lambda': 0.,
                                                      'Hubble_param': 1.0, 'BoxSize': 1.0, 'Sigma8': 1.0},
                                           unitdata={'UnitLength_in_Mpc': 1.0, 'UnitVelocity_in_kms': 1.0,
                                                     'UnitMass_in_Msol': 1.0, 'Flag_physical_comoving': True, 'Flag_hubble_flow': False},
                                           ibuildheadtail=0, ibuildmajormergers=0, TEMPORALHALOIDVAL=1000000000000):
    """
    produces a unifed HDF5 formatted file containing the full catalog plus information to walk the tree
    #ref BuildTemporalHeadTail must have been called before otherwise it is called.
    Code produces a file for each snapshot
    The keys are the same as that contained in the halo catalog dictionary with the addition of
    Num_of_snaps, and similar header info contain in the VELOCIraptor hdf files, ie Num_of_groups, Total_num_of_groups

    #todo don't know if I should use multiprocessing here to write files in parallel. IO might not be ideal

    Here the halodata is the dictionary contains the information

    """

    #Get the snapshot offset if present in the header information
    if("HaloID_snapshot_offset" in rawtree["Header"]):
        snapshotoffset = rawtree["Header"]["HaloID_snapshot_offset"]
    else:
        snapshotoffset = 0

    if (ibuildheadtail == 1):
        BuildTemporalHeadTail(numsnaps, rawtree, numhalos, halodata)
    if (ibuildmajormergers == 1):
        IdentifyMergers(numsnaps, rawtree, numhalos,
                        halodata, cosmodata['BoxSize'], cosmodata['Hubble_param'], atime)
    hdffile = h5py.File(fname+".snap.hdf.data", 'w')
    headergrp = hdffile.create_group("Header")
    # store useful information such as number of snapshots, halos,
    # cosmology (Omega_m, Omega_b, Hubble_param, Omega_Lambda, Box size)
    # units (Physical [1/0] for physical/comoving flag, length in Mpc, km/s, solar masses, Gravity
    # and TEMPORALHALOIDVAL used to traverse tree information (converting halo ids to haloindex or snapshot), Reverse_order [1/0] for last snap listed first)
    # set the attributes of the header
    headergrp.attrs["NSnaps"] = numsnaps
    # overall description
    headergrp.attrs["Title"] = descripdata["Title"]
    # simulation box size
    headergrp.attrs["BoxSize"] = cosmodata["BoxSize"]
    findergrp = headergrp.create_group("HaloFinder")
    findergrp.attrs["Name"] = "VELOCIraptor"
    findergrp.attrs["Version"] = descripdata["VELOCIraptor_version"]
    findergrp.attrs["Particle_num_threshold"] = descripdata["Particle_num_threshold"]

    treebuildergrp = headergrp.create_group("TreeBuilder")
    treebuildergrp.attrs["Name"] = "VELOCIraptor-Tree"
    for field in rawtree["Header"].keys():
        treebuildergrp.attrs[field] = rawtree["Header"][field]

    # cosmological params
    cosmogrp = headergrp.create_group("Cosmology")
    for key in cosmodata.keys():
        if (key != 'BoxSize'):
            cosmogrp.attrs[key] = cosmodata[key]
    # unit params
    unitgrp = headergrp.create_group("Units")
    for key in unitdata.keys():
        unitgrp.attrs[key] = unitdata[key]
    # particle types
    partgrp = headergrp.create_group("Parttypes")
    partgrp.attrs["Flag_gas"] = descripdata["Flag_gas"]
    partgrp.attrs["Flag_star"] = descripdata["Flag_star"]
    partgrp.attrs["Flag_bh"] = descripdata["Flag_bh"]

    # now have finished with header

    # now need to create groups for halos and then a group containing tree information
    snapsgrp = hdffile.create_group("Snapshots")
    # internal tree keys
    treekeys = ["RootHead", "RootHeadSnap", "Head", "HeadSnap",
                "Tail", "TailSnap", "RootTail", "RootTailSnap", "Num_progen"]

    for i in range(snapshotoffset,snapshotoffset+numsnaps):
        # note that I normally have information in reverse order so that might be something in the units
        snapgrp = snapsgrp.create_group("Snap_%03d" % (numsnaps-1-i))
        snapgrp.attrs["Snapnum"] = i
        snapgrp.attrs["NHalos"] = numhalos[i]
        snapgrp.attrs["scalefactor"] = atime[i]
    # now close file and use the pytables interface so as to write the table
    hdffile.close()
    # now write tables using pandas interface
    for i in range(numsnaps):
        # lets see if we can alter the code to write a table
        keys = halodata[i].keys()
        # remove tree keys
        for tkey in treekeys:
            keys.remove(tkey)
        # make temp dict
        dictval = dict()
        for key in keys:
            dictval[key] = halodata[i][key]
        # make a pandas DataFrame using halo dictionary
        df = pd.DataFrame.from_dict(dictval)
        df.to_hdf(fname+".snap.hdf.data", "Snapshots/Snap_%03d/Halos" %
                  (numsnaps-1-i), format='table', mode='a')

    # reopen with h5py interface
    hdffile = h5py.File(fname+".snap.hdf.data", 'a')
    # then write tree information in separate group
    treegrp = hdffile.create_group("MergerTree")
    # Tree group should contain
    """
        HaloSnapID
        HaloSnapNum
        HaloSnapIndex
        ProgenitorIndex
        ProgenitorSnapnum
        ProgenitorID
        DescendantIndex
        ..
        ..
        RootProgenitorIndex
        ..
        ..
        RootDescendantIndex
    """
    # to save on memory, allocate each block separately
    # store halo information
    tothalos = sum(numhalos)
    tdata = np.zeros(tothalos, dtype=halodata[0]["ID"].dtype)
    count = 0
    for i in range(snapshotoffset,snapshotoffset+numsnaps):
        tdata[count:int(numhalos[i])+count] = halodata[i]["ID"]
        count += int(numhalos[i])
    treegrp.create_dataset("HaloSnapID", data=tdata)
    tdata = np.zeros(tothalos, dtype=np.uint32)
    count = 0
    for i in range(snapshotoffset,snapshotoffset+numsnaps):
        tdata[count:int(numhalos[i])+count] = i
        count += int(numhalos[i])
    treegrp.create_dataset("HaloSnapNum", data=tdata)
    tdata = np.zeros(tothalos, dtype=np.uint64)
    count = 0
    for i in range(snapshotoffset,snapshotoffset+numsnaps):
        tdata[count:int(numhalos[i])+count] = range(int(numhalos[i]))
        count += int(numhalos[i])
    treegrp.create_dataset("HaloSnapIndex", data=tdata)
    # store progenitors
    tdata = np.zeros(tothalos, dtype=halodata[0]["Tail"].dtype)
    count = 0
    for i in range(snapshotoffset,snapshotoffset+numsnaps):
        tdata[count:int(numhalos[i])+count] = halodata[i]["Tail"]
        count += int(numhalos[i])
    treegrp.create_dataset("ProgenitorID", data=tdata)
    tdata = np.zeros(tothalos, dtype=halodata[0]["TailSnap"].dtype)
    count = 0
    for i in range(snapshotoffset,snapshotoffset+numsnaps):
        tdata[count:int(numhalos[i])+count] = halodata[i]["TailSnap"]
        count += int(numhalos[i])
    treegrp.create_dataset("ProgenitorSnapnum", data=tdata)
    tdata = np.zeros(tothalos, dtype=np.uint64)
    count = 0
    for i in range(snapshotoffset,snapshotoffset+numsnaps):
        tdata[count:int(numhalos[i])+count] = (halodata[i]
                                               ["Tail"] % TEMPORALHALOIDVAL-1)
        count += int(numhalos[i])
    treegrp.create_dataset("ProgenitorIndex", data=tdata)
    # store descendants
    tdata = np.zeros(tothalos, dtype=halodata[0]["Head"].dtype)
    count = 0
    for i in range(snapshotoffset,snapshotoffset+numsnaps):
        tdata[count:int(numhalos[i])+count] = halodata[i]["Head"]
        count += int(numhalos[i])
    treegrp.create_dataset("DescendantID", data=tdata)
    tdata = np.zeros(tothalos, dtype=halodata[0]["HeadSnap"].dtype)
    count = 0
    for i in range(snapshotoffset,snapshotoffset+numsnaps):
        tdata[count:int(numhalos[i])+count] = halodata[i]["HeadSnap"]
        count += int(numhalos[i])
    treegrp.create_dataset("DescendantSnapnum", data=tdata)
    tdata = np.zeros(tothalos, dtype=np.uint64)
    count = 0
    for i in range(snapshotoffset,snapshotoffset+numsnaps):
        tdata[count:int(numhalos[i])+count] = (halodata[i]
                                               ["Head"] % TEMPORALHALOIDVAL-1)
        count += int(numhalos[i])
    treegrp.create_dataset("DescendantIndex", data=tdata)
    # store progenitors
    tdata = np.zeros(tothalos, dtype=halodata[0]["RootTail"].dtype)
    count = 0
    for i in range(snapshotoffset,snapshotoffset+numsnaps):
        tdata[count:int(numhalos[i])+count] = halodata[i]["RootTail"]
        count += int(numhalos[i])
    treegrp.create_dataset("RootProgenitorID", data=tdata)
    tdata = np.zeros(tothalos, dtype=halodata[0]["RootTailSnap"].dtype)
    count = 0
    for i in range(snapshotoffset,snapshotoffset+numsnaps):
        tdata[count:int(numhalos[i])+count] = halodata[i]["RootTailSnap"]
        count += int(numhalos[i])
    treegrp.create_dataset("RootProgenitorSnapnum", data=tdata)
    tdata = np.zeros(tothalos, dtype=np.uint64)
    count = 0
    for i in range(snapshotoffset,snapshotoffset+numsnaps):
        tdata[count:int(numhalos[i])+count] = (halodata[i]
                                               ["RootTail"] % TEMPORALHALOIDVAL-1)
        count += int(numhalos[i])
    treegrp.create_dataset("RootProgenitorIndex", data=tdata)
    # store descendants
    tdata = np.zeros(tothalos, dtype=halodata[0]["RootHead"].dtype)
    count = 0
    for i in range(snapshotoffset,snapshotoffset+numsnaps):
        tdata[count:int(numhalos[i])+count] = halodata[i]["RootHead"]
        count += int(numhalos[i])
    treegrp.create_dataset("RootDescendantID", data=tdata)
    tdata = np.zeros(tothalos, dtype=halodata[0]["RootHeadSnap"].dtype)
    count = 0
    for i in range(snapshotoffset,snapshotoffset+numsnaps):
        tdata[count:int(numhalos[i])+count] = halodata[i]["RootHeadSnap"]
        count += int(numhalos[i])
    treegrp.create_dataset("RootDescendantSnapnum", data=tdata)
    tdata = np.zeros(tothalos, dtype=np.uint64)
    count = 0
    for i in range(snapshotoffset,snapshotoffset+numsnaps):
        tdata[count:int(numhalos[i])+count] = (halodata[i]
                                               ["RootHead"] % TEMPORALHALOIDVAL-1)
        count += int(numhalos[i])
    treegrp.create_dataset("RootDescendantIndex", data=tdata)
    # store number of progenitors
    tdata = np.zeros(tothalos, dtype=np.uint32)
    count = 0
    for i in range(snapshotoffset,snapshotoffset+numsnaps):
        tdata[count:int(numhalos[i])+count] = halodata[i]["Num_progen"]
        count += int(numhalos[i])
    treegrp.create_dataset("NProgen", data=tdata)

    hdffile.close()


def ReadUnifiedTreeandHaloCatalog(fname, desiredfields=[], iverbose=False, ireversesnaporder=False):
    """
    Read Unified Tree and halo catalog from HDF file with base filename fname.

    Parameters
    ----------

    Returns
    -------
    """
    hdffile = h5py.File(fname, 'r')

    # load data sets containing number of snaps
    headergrpname = "Header/"
    numsnaps = hdffile[headergrpname].attrs["NSnaps"]

    # allocate memory
    halodata = [dict() for i in range(numsnaps)]
    numhalos = [0 for i in range(numsnaps)]
    atime = [0 for i in range(numsnaps)]
    tree = [[] for i in range(numsnaps)]
    simdata = dict()
    unitdata = dict()

    # load simulation (cosmology data
    simgrpname = "Simulation/"
    simgrpname = "Cosmology/"
    fieldnames = [str(n)
                  for n in hdffile[headergrpname+simgrpname].attrs.keys()]
    for fieldname in fieldnames:
        simdata[fieldname] = hdffile[headergrpname +
                                       simgrpname].attrs[fieldname]

    # load unit data
    unitgrpname = "Units/"
    fieldnames = [str(n)
                  for n in hdffile[headergrpname+unitgrpname].attrs.keys()]
    for fieldname in fieldnames:
        unitdata[fieldname] = hdffile[headergrpname +
                                      unitgrpname].attrs[fieldname]

    # for each snap load the appropriate group
    start = time.process_time()
    for i in range(numsnaps):
        if (ireversesnaporder == True):
            snapgrpname = "Snap_%03d/" % (numsnaps-1-i)
        else:
            snapgrpname = "Snap_%03d/" % i
        if (iverbose == True):
            print("Reading ", snapgrpname)
        isnap = hdffile[snapgrpname].attrs["Snapnum"]
        atime[isnap] = hdffile[snapgrpname].attrs["scalefactor"]
        numhalos[isnap] = hdffile[snapgrpname].attrs["NHalos"]
        if (len(desiredfields) > 0):
            fieldnames = desiredfields
        else:
            fieldnames = [str(n) for n in hdffile[snapgrpname].keys()]
        for catvalue in fieldnames:
            halodata[isnap][catvalue] = np.array(
                hdffile[snapgrpname+catvalue])
    hdffile.close()
    print("read halo data ", time.process_time()-start)
    return halodata, numhalos, atime, simdata, unitdata


def WriteWalkableHDFTree(fname, numsnaps, tree, numhalos, halodata, atime,
        descripdata={
        'Title':'Walkable Tree',
        'TreeBuilder' : {
            'Name' : 'TreeFrog',
            'Version' : 1,
            'Temporal_linking_length' : 1,
            'Temporal_halo_id_value' : 1000000000000,
            'Tree_direction': 1,
            'Temporally_Unique_Halo_ID_Description': 'Snap_num*Temporal_linking_length+Index+1'
        },
        'HaloFinder' : {
            'Name' : 'VELOCIraptor',
            'Version' : 1,
            'Particle_num_threshold' : 20,
            'Subhalo_Particle_num_threshold': 20,
            },
        }
        ):
    """
    Produces a HDF5 formatted file containing Reduced Walkable Tree information,
    ie; RootHead, Head, HeadSnap, Tail, RootTail, etc.

    Parameters
    ----------
    fname : string
        filename of the hdf file to be written
    numsnaps : int
        the number of snapshots
    tree : dict
        the tree data
    numhalos : array
        array of number of halos per snapshot
    halodata : dict
        the halo data dictionary
    atime : array
        array of scalefactors/times of the snaphots
    discrptdata : dict
        stores a description of how the tree catalogue was produced

    Returns
    -------
    void :
        Only writes an hdf file. Nothing is returned.
    """
    hdffile = h5py.File(fname, 'w')
    headergrp = hdffile.create_group("Header")
    # set the attributes of the header, store useful information regarding the tree
    headergrp.attrs["NSnaps"] = numsnaps
    # overall description
    headergrp.attrs["Title"] = descripdata["Title"]

    #write all the header info
    for field in tree["Header"]:
        headergrp.attrs[field] = tree["Header"][field]
    treebuildergrp = headergrp.create_group("TreeBuilder")
    for field in descripdata['TreeBuilder']:
        treebuildergrp.attrs[field] = descripdata['TreeBuilder'][field]
    halofindergrp = headergrp.create_group("HaloFinder")
    for field in descripdata['HaloFinder']:
        halofindergrp.attrs[field] = descripdata['HaloFinder'][field]

    # now need to create groups for halos and then a group containing tree information
    snapsgrp = hdffile.create_group("Snapshots")
    # tree keys of interest
    halokeys = ["RootHead", "RootHeadSnap", "RootHeadIndex", "Head", "HeadSnap", "HeadIndex", "Tail",
                "TailSnap", "TailIndex", "RootTail", "RootTailSnap", "RootTailIndex", "ID", "Num_progen"]

    if("HaloID_snapshot_offset" in tree["Header"]):
        snapshotoffset = tree["Header"]["HaloID_snapshot_offset"]
    else:
        snapshotoffset = 0

    for i in range(snapshotoffset,snapshotoffset+numsnaps):
        # note that I normally have information in reverse order so that might be something in the units
        snapgrp = snapsgrp.create_group("Snap_%03d" % i)
        snapgrp.attrs["Snapnum"] = i
        snapgrp.attrs["NHalos"] = numhalos[i]
        snapgrp.attrs["scalefactor"] = atime[i]
        for key in halokeys:
            snapgrp.create_dataset(
                key, data=halodata[i][key], compression="gzip", compression_opts=6)
    hdffile.close()


def ReadWalkableHDFTree(fname, iverbose=True):
    """
    Reads a simple walkable hdf tree file.
    Assumes the input has
    ["RootHead", "RootHeadSnap", "Head", "HeadSnap", "Tail", "TailSnap", "RootTail", "RootTailSnap", "ID", "Num_progen"]
    along with attributes per snap of the scale factor (eventually must generalize to time as well )
    should also have a header gropu with attributes like number of snapshots.
    Returns the halos IDs with walkable tree data, number of snaps, and the number of snapshots searched.
    """
    hdffile = h5py.File(fname, 'r')

    #Read the header information
    numsnaps = hdffile['Header'].attrs["NSnaps"]
    #nsnapsearch = ["Header/TreeBuilder"].attrs["Temporal_linking_length"]

    if (iverbose):
        print("number of snaps", numsnaps)
    treedata = {i:dict() for i in range(numsnaps)}

    treedata["Header"] = dict()
    for field in hdffile["Header"].attrs.keys():
        treedata["Header"][field] = hdffile["Header"].attrs[field]
    treedata["Header"]['TreeBuilder'] = dict()
    for field in hdffile["Header"]['TreeBuilder'].attrs.keys():
        treedata["Header"]['TreeBuilder'][field] = hdffile["Header"]['TreeBuilder'].attrs[field]
    treedata["Header"]['HaloFinder'] = dict()
    for field in hdffile["Header"]['HaloFinder'].attrs.keys():
        treedata["Header"]['HaloFinder'][field] = hdffile["Header"]['HaloFinder'].attrs[field]

    #Get the snapshot offset if present
    if("HaloID_snapshot_offset" in treedata["Header"]):
        snapshotoffset = treedata["Header"]["HaloID_snapshot_offset"]
    else:
        snapshotoffset = 0

    for i in range(snapshotoffset,snapshotoffset+numsnaps):
        # note that I normally have information in reverse order so that might be something in the units
        if (iverbose):
            print("snap ", i)
        for key in hdffile['Snapshots']['Snap_%03d' % i].keys():
            treedata[i][key] = np.array(
                hdffile['Snapshots']['Snap_%03d' % i][key])
    hdffile.close()
    # , nsnapsearch
    return treedata, numsnaps


def FixTruncationBranchSwapsInTreeDescendantAndWrite(rawtreefname, reducedtreename, snapproplistfname, outputupdatedreducedtreename,
                                                     descripdata={'Title': 'Tree catalogue', 'VELOCIraptor_version': 1.3, 'Tree_version': 1.1,
                                                                  'Particle_num_threshold': 20, 'Temporal_linking_length': 1, 'Flag_gas': False, 'Flag_star': False, 'Flag_bh': False},
                                                     npartlim=200, meritlim=0.025, xdifflim=2.0, vdifflim=1.0, nsnapsearch=4,
                                                     searchdepth=2, iswaphalosubhaloflag=1,
                                                     TEMPORALHALOIDVAL=1000000000000, iverbose=1,
                                                     ichecktree = False,
                                                     ibuildtree=False, inputtreeformat=2, inputpropformat=2, inputpropsplitformat=0):
    """
    Updates a tree produced by TreeFrog to correct any branch swap events leading to truncation
    that requires full roothead/root tail information to correctly fix.
    """
    rawtreedata = ReadHaloMergerTreeDescendant(
        rawtreefname, False, inputtreeformat, 0, True)
    # and also extract the description used to make the tree
    numsnaps = len(rawtreedata)
    if (ibuildtree):
        halodata = [dict() for i in range(numsnaps)]
        numhalos = np.zeros(numsnaps, dtype=np.uint64)
        BuildTemporalHeadTailDescendant(
            numsnaps, rawtreedata, numhalos, halodata, TEMPORALHALOIDVAL)
    else:
        halodata = ReadWalkableHDFTree(reducedtreename)

    proplist = ['npart', 'hostHaloID', 'Structuretype', 'ID', 'Xc',
                'Yc', 'Zc', 'VXc', 'VYc', 'VZc', 'Rmax', 'Vmax', 'R_200crit']
    numhalos = np.zeros(numsnaps, dtype=np.uint64)
    atime = np.zeros(numsnaps)
    snaplist = open(snapproplistfname, 'r')
    for i in range(numsnaps):
        snapfile = snaplist.readline().strip()
        halotemp, numhalos[i] = ReadPropertyFile(
            snapfile, inputpropformat, inputpropsplitformat, 0, proplist)
        halodata[i].update(halotemp)
        atime[i]=halodata[i]['SimulationInfo']['ScaleFactor']
    halodata = FixTruncationBranchSwapsInTreeDescendant(numsnaps, rawtreedata, halodata, numhalos,
                                                        npartlim, meritlim, xdifflim, vdifflim, nsnapsearch,
                                                        searchdepth, iswaphalosubhaloflag,
                                                        TEMPORALHALOIDVAL)
    WriteWalkableHDFTree(outputupdatedreducedtreename, numsnaps,
                         rawtreedata, numhalos, halodata, atime, descripdata)
    # return rawtreedata, halodata, numhalos, atime

def FixBranchMergePhaseSearch(numsnaps, treedata, halodata, numhalos,
                            period,
                            npartlim, meritlim, xdifflim, vdifflim, nsnapsearch,
                            TEMPORALHALOIDVAL, iverbose,
                            haloID, haloSnap, haloIndex, haloRootHeadID,
                            mergeHalo, mergeSnap, mergeIndex,
                            premergeHalo, premergeSnap, premergeIndex,
                            postmergeHalo,postmergeSnap, postmergeIndex,
                            secondaryProgenList,
                            iaddsubs=False
                            ):
    """
    Given an object that has fragmented (postmerge)
    look at immediate progenitor and see
    if progenitor or its host has more than on progenitor
    examine these possible candidates looking at phase-space positions
    to identify a possible progenitor for object without a progenitor
    or identify a branch swap where post merge descendant should be swapped

    #todo adding in searching host objects and all subhalos seems to
    generate a broken tree as essentially starting to look across multiple branches
    when trying to patch tree. This is best left for halo tracking codes

    #todo could also check if mergeHalo and object without progenitor are both
    halos and the mergeHalo's descendant is a subhalo of the object
    without progenitor. Then could take over line
    """

    branchfixMerge = branchfixMergeSwapBranch = -1
    minxdiff = minvdiff = minphase2diff = minnpartdiff = 1e32
    haloIDList = []

    # starting at halo that fragments indicating that its progenitors must have
    # merged, move  backwards in time for snapsearch storing
    # the object IDs of the main branch of this object and the any of its
    # hosts.
    curHalo = mergeHalo
    curSnap = np.uint64(curHalo/TEMPORALHALOIDVAL)
    curIndex = np.uint64(curHalo % TEMPORALHALOIDVAL-1)
    curRootHead = halodata[curSnap]['RootHead'][curIndex]
    searchrange = max(0, curSnap-nsnapsearch)

    if (halodata[postmergeSnap]['Num_progen'][postmergeIndex] > 1):
        haloIDList.append(postmergeHalo)
    if (halodata[curSnap]['Num_progen'][curIndex] > 1):
        haloIDList.append(curHalo)
    curHost = halodata[curSnap]['hostHaloID'][curIndex]
    if (curHost != -1):
        curHostSnap = np.uint64(curHost / TEMPORALHALOIDVAL)
        curHostIndex = np.uint64(curHost % TEMPORALHALOIDVAL-1)
        if (halodata[curSnap]['Num_progen'][curHostIndex] > 1 and
            halodata[curSnap]['RootHead'][curHostIndex] == curRootHead):
            haloIDList.append(curHost)
        if (iaddsubs == True):
            subs = np.where((halodata[curSnap]['hostHaloID'] == curHost)*
                            (halodata[curSnap]['Num_progen'][curHostIndex]>1)*
                            (halodata[curSnap]['npart'] >= npartlim)*
                            (halodata[curSnap]['RootHead'] == curRootHead)
                            )[0]
            if (subs.size >0):
                for isub in halodata[curSnap]['ID'][subs]:
                    haloIDList.append(isub)
    # search backwards in time for any object that might have merged with either object or objects host
    while(curSnap >= searchrange and halodata[curSnap]['Tail'][curIndex] != curHalo):
        curHalo = halodata[curSnap]['Tail'][curIndex]
        curSnap = np.uint64(curHalo/TEMPORALHALOIDVAL)
        curIndex = np.uint64(curHalo % TEMPORALHALOIDVAL-1)
        if (halodata[curSnap]['Num_progen'][curIndex] > 1):
            haloIDList.append(curHalo)
        curHost = halodata[curSnap]['hostHaloID'][curIndex]
        if (curHost != -1):
            curHostSnap = np.uint64(curHost/TEMPORALHALOIDVAL)
            curHostIndex = np.uint64(curHost % TEMPORALHALOIDVAL-1)
            if (halodata[curSnap]['Num_progen'][curHostIndex] > 1 and
                halodata[curSnap]['RootHead'][curHostIndex] == curRootHead):
                haloIDList.append(curHost)
            if (iaddsubs == True):
                subs = np.where((halodata[curSnap]['hostHaloID'] == curHost)*
                                (halodata[curSnap]['Num_progen'][curHostIndex]>1)*
                                (halodata[curSnap]['npart'] >= npartlim)*
                                (halodata[curSnap]['RootHead'] == curRootHead)
                                )[0]
                if (subs.size >0):
                    for isub in halodata[curSnap]['ID'][subs]:
                        haloIDList.append(isub)

    # with this list of objects, search if any of these objects have
    # secondary progenitors with high Merit_type
    haloIDList = np.unique(np.array(haloIDList, dtype=np.int64))
    #identify possible secondary progenitors of halos of interest
    mergeCheck = (np.in1d(secondaryProgenList['Descen'], haloIDList) *
            (secondaryProgenList['Merit'] >= meritlim)
            )
    mergeCandidateList = np.where(mergeCheck)[0]
    if (iverbose > 1):
        print('halo in phase check general ', haloID, 'with number of candidates', mergeCandidateList.size,
        'from possible candidates of', haloIDList.size)

    if (mergeCandidateList.size == 0):
        return branchfixMerge, branchfixMergeSwapBranch
    mergeCandidateList = secondaryProgenList['ID'][mergeCandidateList]
    # all these objects are then compared to object missing a progenitor
    # to see if these secondary progenitors match the phase-space
    # position and hence could be a viable progenitor.
    for candidateID in mergeCandidateList:
        # check its position and velocity relative post merge halo
        candidateSnap = np.uint64(candidateID / TEMPORALHALOIDVAL)
        candidateIndex = np.uint64(candidateID % TEMPORALHALOIDVAL - 1)

        xrel = np.array([
                        halodata[candidateSnap]['Xc'][candidateIndex] - halodata[haloSnap]['Xc'][haloIndex],
                        halodata[candidateSnap]['Yc'][candidateIndex] - halodata[haloSnap]['Yc'][haloIndex],
                        halodata[candidateSnap]['Zc'][candidateIndex] - halodata[haloSnap]['Zc'][haloIndex],
                        ])
        xrel[np.where(xrel > 0.5*period)] -= period
        xrel[np.where(xrel < -0.5*period)] += period
        vrel = np.array([
                        halodata[candidateSnap]['VXc'][candidateIndex] - halodata[haloSnap]['VXc'][haloIndex],
                        halodata[candidateSnap]['VYc'][candidateIndex] - halodata[haloSnap]['VYc'][haloIndex],
                        halodata[candidateSnap]['VZc'][candidateIndex] - halodata[haloSnap]['VZc'][haloIndex],
                        ])
        rnorm = 1.0/halodata[haloSnap]['Rmax'][haloIndex]
        vnorm = 1.0/halodata[haloSnap]['Vmax'][haloIndex]
        xdiff = np.linalg.norm(xrel)*rnorm
        vdiff = np.linalg.norm(vrel)*vnorm
        npartdiff = (halodata[candidateSnap]['npart'][candidateIndex]) / float(halodata[haloSnap]['npart'][haloIndex])
        # object must have vdiff < limit and xdiff less than limit to proceed
        if not (xdiff < xdifflim and vdiff < vdifflim):
            continue
        # calculate the phase-difference, min phase-differnce should correspond to candidate progenitor to postmerge line
        phase2diff = xdiff**2.0+vdiff**2.0
        if (phase2diff < minphase2diff and npartdiff > 0.1 and npartdiff < 10.0):
            minphase2diff = phase2diff
            minxdiff = xdiff
            minvdiff = vdiff
            minnpartdiff = npartdiff
            branchfixMerge = np.int64(candidateID)

    if (iverbose > 1):
        oldxdiff, oldvdiff, oldphase2diff, oldnpartdiff = minxdiff, minvdiff, minphase2diff, npartdiff
        oldbranchfixMerge = branchfixMerge

    for candidateID in mergeCandidateList:
        # check its position and velocity relative post merge halo
        candidateSnap = np.uint64(candidateID / TEMPORALHALOIDVAL)
        candidateIndex = np.uint64(candidateID % TEMPORALHALOIDVAL - 1)

        xrel = np.array([
                        halodata[candidateSnap]['Xc'][candidateIndex] - halodata[postmergeSnap]['Xc'][postmergeIndex],
                        halodata[candidateSnap]['Yc'][candidateIndex] - halodata[postmergeSnap]['Yc'][postmergeIndex],
                        halodata[candidateSnap]['Zc'][candidateIndex] - halodata[postmergeSnap]['Zc'][postmergeIndex],
                        ])
        xrel[np.where(xrel > 0.5*period)] -= period
        xrel[np.where(xrel < -0.5*period)] += period
        vrel = np.array([
                        halodata[candidateSnap]['VXc'][candidateIndex] - halodata[postmergeSnap]['VXc'][postmergeIndex],
                        halodata[candidateSnap]['VYc'][candidateIndex] - halodata[postmergeSnap]['VYc'][postmergeIndex],
                        halodata[candidateSnap]['VZc'][candidateIndex] - halodata[postmergeSnap]['VZc'][postmergeIndex],
                        ])
        rnorm = 1.0/halodata[postmergeSnap]['Rmax'][postmergeIndex]
        vnorm = 1.0/halodata[postmergeSnap]['Vmax'][postmergeIndex]
        xdiff = np.linalg.norm(xrel)*rnorm
        vdiff = np.linalg.norm(vrel)*vnorm
        npartdiff = (halodata[candidateSnap]['npart'][candidateIndex]) / float(halodata[postmergeSnap]['npart'][postmergeIndex])

        # object must have vdiff < limit and xdiff less than limit to proceed
        if not (xdiff < xdifflim and vdiff < vdifflim):
            continue
        # calculate the phase-difference, min phase-differnce should correspond to candidate progenitor to postmerge line
        phase2diff = xdiff**2.0+vdiff**2.0
        if (phase2diff < minphase2diff and npartdiff > 0.1 and npartdiff < 10.0):
            minphase2diff = phase2diff
            minxdiff = xdiff
            minvdiff = vdiff
            minnpartdiff = npartdiff
            branchfixMergeSwapBranch = np.int64(candidateID)
            branchfixMerge = -1

    if (iverbose > 1):
        if (branchfixMerge != -1):
            print('Simple halo merge/fragmentation correction. Halos involved ',
            haloID, branchfixMerge,
            'comparison of phase', minxdiff, minvdiff, minnpartdiff)
        if (branchfixMergeSwapBranch != -1):
            print('Branch swapping merge/fragmentation correction. Halos involved',
            haloID, postmergeHalo, branchfixMergeSwapBranch,
            'comparison of phase', minxdiff, minvdiff, minnpartdiff,
            'with old simple merge ', oldbranchfixMerge,
            'comparison of phase', oldxdiff, oldvdiff, oldnpartdiff)

    return branchfixMerge, branchfixMergeSwapBranch

def FixBranchPhaseMergeAdjustTree(numsnaps, treedata, halodata, numhalos,
                            TEMPORALHALOIDVAL, iverbose,
                            haloID, haloSnap, haloIndex,
                            mergeHalo, mergeSnap, mergeIndex,
                            premergeHalo, premergeSnap, premergeIndex,
                            postmergeHalo,postmergeSnap, postmergeIndex,
                            branchfixHalo
                            ):
    """
    Adjust halo merger tree by pointing merged halo to object without progenitor
    by #ref FixBranchMergePhaseSearch
    """

    if (iverbose > 1):
        print('Adjusting simple merge/fragmentation ',haloID, branchfixHalo)
    # store necessary tree points
    branchfixSnap = np.uint64(branchfixHalo / TEMPORALHALOIDVAL)
    branchfixIndex = np.uint64(branchfixHalo % TEMPORALHALOIDVAL - 1)
    newroottailbranchfix = halodata[branchfixSnap]['RootTail'][branchfixIndex]
    newroottailbranchfixSnap = np.uint64(newroottailbranchfix / TEMPORALHALOIDVAL)
    newroottailbranchfixIndex = np.uint64(newroottailbranchfix % TEMPORALHALOIDVAL - 1)

    # adjust branch fix object to point to halo without progenitor
    halodata[branchfixSnap]['Head'][branchfixIndex] = haloID
    halodata[branchfixSnap]['HeadSnap'][branchfixIndex] = haloSnap

    # point object without progenitor to branch fix
    halodata[haloSnap]['Tail'][haloIndex] = branchfixHalo
    halodata[haloSnap]['TailSnap'][haloIndex] = branchfixSnap
    halodata[haloSnap]['RootTail'][haloIndex] = newroottailbranchfix
    halodata[haloSnap]['RootTailSnap'][haloIndex] = newroottailbranchfixSnap

    # update the root tails
    curHalo = haloID
    curSnap = np.uint64(curHalo / TEMPORALHALOIDVAL)
    curIndex = np.uint64(curHalo % TEMPORALHALOIDVAL - 1)
    while (True):
        if (iverbose > 2):
            print('moving up branch to adjust the root tails', curHalo,
                  curSnap, halodata[curSnap]['RootTail'][curIndex], newroottailbranchfix)
        halodata[curSnap]['RootTail'][curIndex] = newroottailbranchfix
        halodata[curSnap]['RootTailSnap'][curIndex] = newroottailbranchfixSnap
        # if not on main branch exit
        if (halodata[np.uint32(halodata[curSnap]['Head'][curIndex]/TEMPORALHALOIDVAL)]['Tail'][np.uint64(halodata[curSnap]['Head'][curIndex] % TEMPORALHALOIDVAL-1)] != curHalo):
            break
        # if at root head then exit
        if (halodata[curSnap]['Head'][curIndex] == curHalo):
            break
        curHalo = halodata[curSnap]['Head'][curIndex]
        curSnap = np.uint64(curHalo / TEMPORALHALOIDVAL)
        curIndex = np.uint64(curHalo % TEMPORALHALOIDVAL - 1)

def FixBranchPhaseBranchSwapAdjustTree(numsnaps, treedata, halodata, numhalos,
                            TEMPORALHALOIDVAL, iverbose,
                            haloID, haloSnap, haloIndex,
                            mergeHalo, mergeSnap, mergeIndex,
                            premergeHalo, premergeSnap, premergeIndex,
                            postmergeHalo,postmergeSnap, postmergeIndex,
                            branchfixHalo
                            ):
    """
    Adjust halo merger tree, swapping head tail information for objects found
    by #ref FixBranchMergePhaseSearch
    Specifically have branch fix halo point to post merge halo
    and have merge halo point to no progenitor halo
    """

    if (iverbose > 1):
        print('Adjusting branch swap merge/fragmentation ',haloID, postmergeHalo, branchfixHalo)

    # store the branch fix points
    branchfixSnap = np.uint64(branchfixHalo / TEMPORALHALOIDVAL)
    branchfixIndex = np.uint64(branchfixHalo % TEMPORALHALOIDVAL - 1)
    branchfixHead = halodata[branchfixSnap]['Head'][branchfixIndex]
    branchfixHeadSnap = np.uint64(branchfixHead / TEMPORALHALOIDVAL)
    branchfixHeadIndex = np.uint64(branchfixHead % TEMPORALHALOIDVAL - 1)
    branchfixRootTail = halodata[branchfixSnap]['RootTail'][branchfixIndex]
    branchfixRootTailSnap = np.uint64(branchfixRootTail / TEMPORALHALOIDVAL)
    branchfixRootTailIndex = np.uint64(branchfixRootTail % TEMPORALHALOIDVAL - 1)
    branchfixHeadTail = halodata[branchfixHeadSnap]['Tail'][branchfixHeadIndex]
    branchfixHeadTailSnap = np.uint64(branchfixHeadTail / TEMPORALHALOIDVAL)
    branchfixHeadTailIndex = np.uint64(branchfixHeadTail % TEMPORALHALOIDVAL - 1)
    branchfixHeadTailRootTail = halodata[branchfixHeadTailSnap]['RootTail'][branchfixHeadTailIndex]
    branchfixHeadTailRootTailSnap = np.uint64(branchfixHeadTailRootTail / TEMPORALHALOIDVAL)
    branchfixHeadTailRootTailIndex = np.uint64(branchfixHeadTailRootTail % TEMPORALHALOIDVAL - 1)

    #adjust heads
    halodata[branchfixSnap]['Head'][branchfixIndex] = postmergeHalo
    halodata[branchfixSnap]['HeadSnap'][branchfixIndex] = postmergeSnap

    halodata[mergeSnap]['Head'][mergeIndex] = haloID
    halodata[mergeSnap]['HeadSnap'][mergeIndex] = haloSnap

    #adjust tails
    halodata[postmergeSnap]['Tail'][postmergeIndex] = branchfixHalo
    halodata[postmergeSnap]['TailSnap'][postmergeIndex] = branchfixSnap

    halodata[haloSnap]['Tail'][haloIndex] = mergeHalo
    halodata[haloSnap]['TailSnap'][haloIndex] = mergeSnap

    #adjust root tails
    curHalo = haloID
    curSnap = np.uint64(curHalo / TEMPORALHALOIDVAL)
    curIndex = np.uint64(curHalo % TEMPORALHALOIDVAL - 1)
    while (True):
        if (iverbose > 2):
            print('moving up branch to adjust the root tails', curHalo,
                  curSnap, halodata[curSnap]['RootTail'][curIndex], branchfixHeadTailRootTail)
        halodata[curSnap]['RootTail'][curIndex] = branchfixHeadTailRootTail
        halodata[curSnap]['RootTailSnap'][curIndex] = branchfixHeadTailRootTailSnap
        # if not on main branch exit
        if (halodata[np.uint32(halodata[curSnap]['Head'][curIndex]/TEMPORALHALOIDVAL)]['Tail'][np.uint64(halodata[curSnap]['Head'][curIndex] % TEMPORALHALOIDVAL-1)] != curHalo):
            break
        # if at root head then exit
        if (halodata[curSnap]['Head'][curIndex] == curHalo):
            break
        curHalo = halodata[curSnap]['Head'][curIndex]
        curSnap = np.uint64(curHalo / TEMPORALHALOIDVAL)
        curIndex = np.uint64(curHalo % TEMPORALHALOIDVAL - 1)

    curHalo = postmergeHalo
    curSnap = np.uint64(curHalo / TEMPORALHALOIDVAL)
    curIndex = np.uint64(curHalo % TEMPORALHALOIDVAL - 1)
    while (True):
        if (iverbose > 2):
            print('moving up fix branch to adjust the root tails', curHalo, curSnap,
                  halodata[curSnap]['RootTail'][curIndex], branchfixRootTail)
        halodata[curSnap]['RootTail'][curIndex] = branchfixRootTail
        halodata[curSnap]['RootTailSnap'][curIndex] = branchfixRootTailSnap
        # if not on main branch exit
        if (halodata[np.uint32(halodata[curSnap]['Head'][curIndex]/TEMPORALHALOIDVAL)]['Tail'][np.uint64(halodata[curSnap]['Head'][curIndex] % TEMPORALHALOIDVAL-1)] != curHalo):
            break
        # if at root head then exit
        if (halodata[curSnap]['Head'][curIndex] == curHalo):
            break
        curHalo = halodata[curSnap]['Head'][curIndex]
        curSnap = np.uint64(curHalo/TEMPORALHALOIDVAL)
        curIndex = np.uint64(curHalo % TEMPORALHALOIDVAL-1)

def FixBranchHaloSubhaloSwapBranch(numsnaps, treedata, halodata, numhalos,
                            period,
                            npartlim, meritlim, xdifflim, vdifflim, nsnapsearch,
                            TEMPORALHALOIDVAL, iverbose,
                            haloID, haloSnap, haloIndex, haloRootHeadID,
                            mergeHalo,
                            secondaryProgenList
                            ):
    """
    Given either a halo or a subhalo with no progenitor, see if tree
    can be adjusted.
    For halo without progenitor, see if one of its subhalos can be left stranded in the tree
    and have halo take over subhalo's merger tree history (main branch and associated secondary branches)
    For subhalo without a progenitor but with a main branch descendant line, see if
    host halo of subhalo terminates as secondary branch of subhalo main branch and adjust
    halo to take the merger tree main branch (and secondary branches) of the subhalo.
    """
    branchfixSwapBranch = -1
    branchfixSwapBranchSubhalo = -1
    branchfixSwapBranchHead = -1
    branchfixSwapBranchTail = -1
    branchfixMerit = -1
    branchfixNpart = 0
    minxdiff = minvdiff = minphase2diff = minnpartdiff = 1e32

    branchfixSwapBranchMeritIndex = -1
    mergeHeadHost = -1
    if (halodata[haloSnap]['hostHaloID'][haloIndex] == -1):
        haloHost = halodata[haloSnap]['ID'][haloIndex]
        # store halo head and its root tail
        haloHead = halodata[haloSnap]['Head'][haloIndex]
        haloHeadSnap = np.uint64(haloHead / TEMPORALHALOIDVAL)
        haloHeadIndex = np.uint64(haloHead % TEMPORALHALOIDVAL-1)
        haloHeadRootTail = halodata[haloHeadSnap]['RootTail'][haloHeadIndex]

        # find subhalos that have the same root descendant
        # ideally would like to expand search to also use mergeHalo but also could alter phase-search to
        # for halo's to take over mergeHalo's descendant line if that mergeHalo is a subhalo
        subs = np.where((halodata[haloSnap]['hostHaloID'] == haloHost) *
            (halodata[haloSnap]['RootHead'] == haloRootHeadID)
            )[0]
        if (iverbose > 1):
            print(haloID, halodata[haloSnap]['npart'][haloIndex], mergeHalo, mergeHeadHost, 'halo has subhalos ', halodata[haloSnap]['ID'][subs],
              'composed of npart', halodata[haloSnap]['npart'][subs],
              'of type', halodata[haloSnap]['Structuretype'][subs],
              'that might have progenitors and descendants that could match host',
              )

        if (subs.size > 0):
            # with subhalo(s) of interest check to see if their progenitors point to halo a secdonary descendant
            for isub in subs:
                subTail = halodata[haloSnap]['Tail'][isub]
                subTailSnap = np.uint64(subTail / TEMPORALHALOIDVAL)
                subTailIndex = np.uint64(subTail % TEMPORALHALOIDVAL-1)
                if (iverbose > 1):
                    print('sub candidate', halodata[haloSnap]['ID'][isub], 'with progenitor', subTail,
                      halodata[subTailSnap]['RootTail'][subTailIndex],
                      treedata[subTailSnap]['Num_descen'][subTailIndex],
                      treedata[subTailSnap]['Merit'][subTailIndex],
                      treedata[subTailSnap]['Descen'][subTailIndex],
                    )
                if (treedata[subTailSnap]['Num_descen'][subTailIndex] == 1):
                    continue
                wdata = np.where(treedata[subTailSnap]['Descen'][subTailIndex] == haloID)[0]
                if (wdata.size == 0):
                    continue
                if (treedata[subTailSnap]['Descen'][subTailIndex][wdata] == haloID and
                    #treedata[subTailSnap]['Merit'][subTailIndex][wdata] >= meritlim and
                    treedata[subTailSnap]['Merit'][subTailIndex][wdata] > branchfixMerit):
                    branchfixSwapBranch = halodata[haloSnap]['ID'][isub]
                    branchfixSwapBranchTail = subTail
                    branchfixMerit = treedata[subTailSnap]['Merit'][subTailIndex][wdata][0]
                    branchfixSwapBranchMeritIndex = wdata[0]
            # have found the largest merit. Either proceed to with swap if merit is above merit limit
            # or if npart ratio and phase-space difference small enough, forcing halo line to cotinue
            if (branchfixMerit < meritlim and branchfixSwapBranch != -1):
                branchfixSwapBranchSnap = np.uint64(branchfixSwapBranch / TEMPORALHALOIDVAL)
                branchfixSwapBranchIndex = np.uint64(branchfixSwapBranch % TEMPORALHALOIDVAL-1)
                branchfixSwapBranchTailSnap = np.uint64(branchfixSwapBranchTail / TEMPORALHALOIDVAL)
                branchfixSwapBranchTailIndex = np.uint64(branchfixSwapBranchTail % TEMPORALHALOIDVAL-1)
                xrel = np.array([
                                halodata[haloSnap]['Xc'][haloIndex] - halodata[branchfixSwapBranchTailSnap]['Xc'][branchfixSwapBranchTailIndex],
                                halodata[haloSnap]['Yc'][haloIndex] - halodata[branchfixSwapBranchTailSnap]['Yc'][branchfixSwapBranchTailIndex],
                                halodata[haloSnap]['Zc'][haloIndex] - halodata[branchfixSwapBranchTailSnap]['Zc'][branchfixSwapBranchTailIndex],
                                ])
                xrel[np.where(xrel > 0.5*period)] -= period
                xrel[np.where(xrel < -0.5*period)] += period
                vrel = np.array([
                                halodata[haloSnap]['VXc'][haloIndex] - halodata[branchfixSwapBranchTailSnap]['VXc'][branchfixSwapBranchTailIndex],
                                halodata[haloSnap]['VYc'][haloIndex] - halodata[branchfixSwapBranchTailSnap]['VYc'][branchfixSwapBranchTailIndex],
                                halodata[haloSnap]['VZc'][haloIndex] - halodata[branchfixSwapBranchTailSnap]['VZc'][branchfixSwapBranchTailIndex],
                                ])

                xrel = np.array([
                                halodata[haloSnap]['Xc'][haloIndex] - halodata[branchfixSwapBranchSnap]['Xc'][branchfixSwapBranchIndex],
                                halodata[haloSnap]['Yc'][haloIndex] - halodata[branchfixSwapBranchSnap]['Yc'][branchfixSwapBranchIndex],
                                halodata[haloSnap]['Zc'][haloIndex] - halodata[branchfixSwapBranchSnap]['Zc'][branchfixSwapBranchIndex],
                                ])
                xrel[np.where(xrel > 0.5*period)] -= period
                xrel[np.where(xrel < -0.5*period)] += period
                vrel = np.array([
                                halodata[haloSnap]['VXc'][haloIndex] - halodata[branchfixSwapBranchSnap]['VXc'][branchfixSwapBranchIndex],
                                halodata[haloSnap]['VYc'][haloIndex] - halodata[branchfixSwapBranchSnap]['VYc'][branchfixSwapBranchIndex],
                                halodata[haloSnap]['VZc'][haloIndex] - halodata[branchfixSwapBranchSnap]['VZc'][branchfixSwapBranchIndex],
                                ])
                rnorm = 1.0/halodata[haloSnap]['Rmax'][haloIndex]
                vnorm = 1.0/halodata[haloSnap]['Vmax'][haloIndex]
                xdiff = np.linalg.norm(xrel)*rnorm
                vdiff = np.linalg.norm(vrel)*vnorm
                npartdiff = (halodata[haloSnap]['npart'][haloIndex]) / float(halodata[branchfixSwapBranchTailSnap]['npart'][branchfixSwapBranchTailIndex])
                npartdiff = (halodata[haloSnap]['npart'][haloIndex]) / float(halodata[branchfixSwapBranchSnap]['npart'][branchfixSwapBranchIndex])
                if (npartdiff > 10.0 or npartdiff < 0.1 and xdiff > xdifflim and vdiff > vdifflim):
                    branchfixSwapBranch = -1
                    branchfixSwapBranchTail = -1

            if (iverbose > 1 and branchfixSwapBranch != -1):
                print(haloID, 'halo has taken subhalo main branch of ', branchfixSwapBranch,
                      'with progenitor of ',branchfixSwapBranchTail,
                      'having npart, stype and root tail of ', halodata[subTailSnap]['npart'][subTailIndex],
                      halodata[subTailSnap]['Structuretype'][subTailIndex], halodata[subTailSnap]['RootTail'][subTailIndex],
                      #'and phase-diff values (if calculated as low merit)', xdiff, vdiff, npartdiff
                )
        else :
            branchfixSwapBranch = -2
    # if object is subhalo and host halo mergers with subhalo branch, fix host halo to take over
    # subhalo branch line
    else:
        mergeHaloHead = -1
        if (mergeHalo != -1):
            mergeSnap = np.uint64(mergeHalo / TEMPORALHALOIDVAL)
            mergeIndex = np.uint64(mergeHalo % TEMPORALHALOIDVAL - 1)
            mergeHaloHead = halodata[mergeSnap]['Head'][mergeIndex]
        haloHost = halodata[haloSnap]['hostHaloID'][haloIndex]
        haloHostIndex = np.uint64(haloHost % TEMPORALHALOIDVAL - 1)
        haloHostSnap = haloSnap
        haloHostRootHeadID = halodata[haloSnap]['RootHead'][haloHostIndex]
        haloHostHead = halodata[haloSnap]['Head'][haloHostIndex]
        haloHostHeadSnap = np.uint64(haloHostHead / TEMPORALHALOIDVAL)
        haloHostHeadIndex = np.uint64(haloHostHead % TEMPORALHALOIDVAL - 1)
        haloHostHeadTail = halodata[haloHostHeadSnap]['Tail'][haloHostHeadIndex]
        haloHostHeadRootTail = halodata[haloHostHeadSnap]['RootTail'][haloHostHeadIndex]

        if (iverbose > 1):
            print('Subhalo', haloID, 'with',
              'npart', halodata[haloSnap]['npart'][haloIndex], halodata[haloSnap]['npart'][haloHostIndex],
              'Stype', halodata[haloSnap]['Structuretype'][haloIndex],
              'has host', haloHost,
              'objects end up as',haloRootHeadID, haloHostRootHeadID)
        # update to adjust subhalo if halo mergers into subhalo branch
        # giving halo the subhalo's descendant branch
        if (haloHostRootHeadID == haloRootHeadID and
            haloHostHeadRootTail == haloID):
            branchfixSwapBranch = haloHost
            branchfixSwapBranchSubhalo = haloID

            if (iverbose > 1):
                print('Subhalo swaping descendant line with halo that mergers immediately',  branchfixSwapBranchSubhalo, branchfixSwapBranch)
        #if halo does not immediately merge then continue search
        elif (haloHostHeadRootTail != haloID and haloHostRootHeadID == haloRootHeadID):
            # move along host halo line to see if it merges with subhalo line within nsnapsearch
            # and store point at which host was last a halo
            curHalo = haloHost
            curSnap = np.uint64(curHalo / TEMPORALHALOIDVAL)
            curIndex = np.uint64(curHalo % TEMPORALHALOIDVAL - 1)
            curRootTail = halodata[curSnap]['RootTail'][curIndex]
            curRootHead = halodata[curSnap]['RootHead'][curIndex]
            haloHostRootTail = halodata[curSnap]['RootTail'][curIndex]
            fixHalo = curHalo
            fixSnap = curSnap
            fixTail = halodata[curSnap]['Tail'][curIndex]
            fixLastAsHalo = curHalo
            fixFirstAsSubhalo = -1
            fixHostTail = fixTail
            #more forward one step
            curHalo = halodata[curSnap]['Head'][curIndex]
            curSnap = np.uint64(curHalo / TEMPORALHALOIDVAL)
            curIndex = np.uint64(curHalo % TEMPORALHALOIDVAL - 1)
            curRootTail = halodata[curSnap]['RootTail'][curIndex]
            curRootHead = halodata[curSnap]['RootHead'][curIndex]
            fixHead = curHalo
            fixHeadSnap = curSnap
            fixRootTail = curRootTail
            ncount = 0
            while (curRootTail != haloID and ncount < nsnapsearch):
                if (halodata[curSnap]['hostHaloID'][curIndex] != -1 and fixFirstAsSubhalo == -1):
                    fixFirstAsSubhalo = curHalo
                    fixLastAsHalo = fixHalo
                fixTail = fixHalo
                fixHalo = curHalo
                fixSnap = curSnap
                ncount += 1
                if (curHalo == curRootHead):
                    break
                curHalo = halodata[curSnap]['Head'][curIndex]
                curSnap = np.uint64(curHalo / TEMPORALHALOIDVAL)
                curIndex = np.uint64(curHalo % TEMPORALHALOIDVAL - 1)
                curRootTail = halodata[curSnap]['RootTail'][curIndex]
                fixHead = curHalo
                fixHeadSnap = curSnap
                fixRootTail = curRootTail
            # if viable merge point has been found, proceed
            if (iverbose > 1):
                print('host at',fixHalo,ncount,'stops being halo at ',fixLastAsHalo, 'in', fixRootTail, haloID)
            if (ncount < nsnapsearch and fixRootTail == haloID):
                # find if/when subhalo becomes a halo
                curHalo = haloID
                curSnap = np.uint64(curHalo / TEMPORALHALOIDVAL)
                curIndex = np.uint64(curHalo % TEMPORALHALOIDVAL - 1)
                subhaloAsSubhalo = curHalo
                subhaloAsSubhaloSnap = curSnap
                subhaloAsSubhaloIndex = curIndex
                subhaloAsHalo = -1
                curHalo = halodata[curSnap]['Head'][curIndex]
                curSnap = np.uint64(curHalo / TEMPORALHALOIDVAL)
                curIndex = np.uint64(curHalo % TEMPORALHALOIDVAL - 1)
                curRootTail = halodata[curSnap]['RootTail'][curIndex]
                curRootHead = halodata[curSnap]['RootHead'][curIndex]
                subhaloAsSubhaloHead = curHalo
                while (curRootTail == haloID and curSnap <= fixHeadSnap):
                    if (halodata[curSnap]['hostHaloID'][curIndex] == -1):
                        subhaloAsHalo = curHalo
                        break
                    if (curHalo == curRootHead):
                        break
                    subhaloAsSubhalo = curHalo
                    subhaloAsSubhaloSnap = curSnap
                    subhaloAsSubhaloIndex = curIndex
                    curHalo = halodata[curSnap]['Head'][curIndex]
                    curSnap = np.uint64(curHalo / TEMPORALHALOIDVAL)
                    curIndex = np.uint64(curHalo % TEMPORALHALOIDVAL - 1)
                    curRootTail = halodata[curSnap]['RootTail'][curIndex]
                    subhaloAsSubhaloHead = curHalo

                # see if host halo ever became a subhalo before it merged.
                if (fixFirstAsSubhalo == -1):
                    branchfixSwapBranchSubhalo = subhaloAsSubhalo
                    branchfixSwapBranch = fixHalo
                    #store head of subhalo
                    branchfixSwapBranchTail = subhaloAsSubhaloHead
                    if (iverbose > 1):
                        print('Subhalo swapping descendant line with halo that has delayed mergers with subhalo main branch',  branchfixSwapBranchSubhalo, branchfixSwapBranch)
                # if subhalo becomes halo and halo swaps to subhalo then
                # adjust line to swap branchs so halo -> halo, subhalo -> subhalo
                elif (fixRootTail == haloID and subhaloAsHalo != -1 and fixFirstAsSubhalo != -1):
                    fixLastAsHaloSnap = np.uint64(fixLastAsHalo / TEMPORALHALOIDVAL)
                    subhaloAsHaloSnap = np.uint64(subhaloAsHalo / TEMPORALHALOIDVAL)
                    if (subhaloAsHaloSnap > fixLastAsHaloSnap):
                        branchfixSwapBranchSubhalo = subhaloAsSubhalo
                        branchfixSwapBranch = fixLastAsHalo
                        branchfixSwapBranchHead = fixFirstAsSubhalo
                        branchfixSwapBranchTail = subhaloAsSubhaloHead
                        if (iverbose > 1):
                            print('Subhalo swapping descendant line with halo that has delayed mergers with subhalo main branch and swapping with subhalo progenitor',
                                    branchfixSwapBranchSubhalo, branchfixSwapBranch, branchfixSwapBranchHead)


            #if still can't find fix, look at when subahlo first becomes halo and see if that halo has any progenitor objects that are also halos
            else:
                curHalo = haloID
                subhaloAsSubhalo = curHalo
                subhaloAsHalo = -1
                curSnap = np.uint64(curHalo / TEMPORALHALOIDVAL)
                curIndex = np.uint64(curHalo % TEMPORALHALOIDVAL - 1)
                curRootTail = halodata[curSnap]['RootTail'][curIndex]
                curRootHead = halodata[curSnap]['RootHead'][curIndex]
                subhaloAsSubhaloHead = curHalo
                ncount = 0
                while (curRootTail == haloID and halodata[curSnap]['hostHaloID'][curIndex] != -1
                    and halodata[curSnap]['RootHead'][curIndex] != curHalo and ncount <nsnapsearch):
                    ncount += 1
                    subhaloAsSubhalo = curHalo
                    curHalo = halodata[curSnap]['Head'][curIndex]
                    curSnap = np.uint64(curHalo / TEMPORALHALOIDVAL)
                    curIndex = np.uint64(curHalo % TEMPORALHALOIDVAL - 1)
                    curRootTail = halodata[curSnap]['RootTail'][curIndex]
                    if (subhaloAsHalo == -1 and halodata[curSnap]['hostHaloID'][curIndex] == -1):
                        subhaloAsHalo = curHalo
                if (halodata[curSnap]['hostHaloID'][curIndex] == -1):
                    candidates = np.where((secondaryProgenList['Descen']==curHalo)*
                        (secondaryProgenList['Structuretype']==10)*
                        (secondaryProgenList['Merit']>=meritlim))[0]
                    if (candidates.size > 0):
                        candidates = candidates[np.argsort(secondaryProgenList['Merit'][candidates])]
                        candidate = np.int64(secondaryProgenList['ID'][candidates[0]])
                        candidateIndex = np.int64(np.int64(secondaryProgenList['ID'][candidates]) / TEMPORALHALOIDVAL)
                        candidateSnap = np.int64(np.int64(secondaryProgenList['ID'][candidates]) % TEMPORALHALOIDVAL - 1)
                        if (iverbose > 1):
                            print('new walk up subhalo till halo',haloID,curHalo,'and have candidate',candidate)
                        branchfixSwapBranchSubhalo = subhaloAsSubhalo
                        branchfixSwapBranch = candidate
                        branchfixSwapBranchTail = subhaloAsHalo


        else :
            branchfixSwapBranch = -2
            if (iverbose > 1):
                print('Subhalo swap not possible, no viable host with same root head')

    return branchfixSwapBranch, branchfixSwapBranchSubhalo, branchfixSwapBranchHead, branchfixSwapBranchTail


def FixBranchHaloSubhaloSwapBranchAdjustTree(numsnaps, treedata, halodata, numhalos,
                            nsnapsearch,
                            TEMPORALHALOIDVAL, iverbose,
                            haloID, haloSnap, haloIndex, haloRootHeadID,
                            branchfixSwapBranch, branchfixSwapBranchSubhalo, branchfixSwapBranchHead, branchfixSwapBranchTail
                            ):
    """
    Adjust tree of halo or subhalo identified by #ref FixBranchCheckReAssignMainBranch
    """
    branchfixIndex = np.uint64(branchfixSwapBranch % TEMPORALHALOIDVAL-1)
    branchfixSnap = np.uint64(branchfixSwapBranch / TEMPORALHALOIDVAL)
    branchfixHead = halodata[branchfixSnap]['Head'][branchfixIndex]
    branchfixHeadSnap = np.uint64(branchfixHead/TEMPORALHALOIDVAL)
    branchfixHeadIndex = np.uint64(branchfixHead % TEMPORALHALOIDVAL-1)
    branchfixTail = halodata[branchfixSnap]['Tail'][branchfixIndex]
    branchfixTailSnap = np.uint64(branchfixTail / TEMPORALHALOIDVAL)
    branchfixTailIndex = np.uint64(branchfixTail % TEMPORALHALOIDVAL-1)
    branchfixRootTail = halodata[branchfixSnap]['RootTail'][branchfixIndex]
    branchfixRootTailSnap = np.uint64(branchfixRootTail / TEMPORALHALOIDVAL)
    branchfixRootTailIndex = np.uint64(branchfixRootTail % TEMPORALHALOIDVAL-1)
    searchrange = max(0, branchfixSnap-nsnapsearch)
    if (halodata[haloSnap]['hostHaloID'][haloIndex] == -1):
        if (iverbose>1):
            print('halo ', haloID, 'now should have progenitor', branchfixTail,
                  'taking over subhalo branch of', branchfixSwapBranch)

        # take subhalo history
        halodata[haloSnap]['Tail'][haloIndex] = branchfixTail
        halodata[haloSnap]['TailSnap'][haloIndex] = branchfixTailSnap
        halodata[haloSnap]['RootTail'][haloIndex] = branchfixRootTail
        halodata[haloSnap]['RootTailSnap'][haloIndex] = branchfixRootTailSnap
        # subhalo ends its history at itself
        halodata[branchfixSnap]['Tail'][branchfixIndex] = branchfixSwapBranch
        halodata[branchfixSnap]['TailSnap'][branchfixIndex] = branchfixSnap
        halodata[branchfixSnap]['RootTail'][branchfixIndex] = branchfixSwapBranch
        halodata[branchfixSnap]['RootTailSnap'][branchfixIndex] = branchfixSnap

        # lets find first instance of when halo either mergers or becomes a subhalo. If that is shorter than
        # the life span of the subhalo it is taking over, then take over subhalo descendant line from that point
        if (halodata[branchfixSnap]['Head'][branchfixIndex] == halodata[haloSnap]['Head'][haloIndex]):
            return
        haloasmainbranchorhostlength = subhalomainbranchlength = 0
        curHalo = haloID
        curSnap = np.uint64(curHalo / TEMPORALHALOIDVAL)
        curIndex = np.uint64(curHalo % TEMPORALHALOIDVAL - 1)
        curHead = halodata[curSnap]['Head'][curIndex]
        RootTail = curRootTail = halodata[curSnap]['RootTail'][curIndex]
        while (curRootTail == RootTail and halodata[curSnap]['hostHaloID'][curIndex] == -1):
            haloasmainbranchorhostlength += 1
            haloEnd = curHalo
            haloEndSnap = curSnap
            haloEndIndex = curIndex
            if (curHalo == curHead):
                break
            curHalo = curHead
            curSnap = np.uint64(curHalo / TEMPORALHALOIDVAL)
            curIndex = np.uint64(curHalo % TEMPORALHALOIDVAL - 1)
            curRootTail = halodata[curSnap]['RootTail'][curIndex]
            curHead = halodata[curSnap]['Head'][curIndex]

        curHalo = branchfixSwapBranch
        curSnap = np.uint64(curHalo / TEMPORALHALOIDVAL)
        curIndex = np.uint64(curHalo % TEMPORALHALOIDVAL - 1)
        curHead = halodata[curSnap]['Head'][curIndex]
        RootTail = curRootTail = halodata[curSnap]['RootTail'][curIndex]
        while (curRootTail == RootTail):
            subhalomainbranchlength += 1
            subhaloEnd = curHalo
            subhaloEndSnap = curSnap
            subhaloEndIndex = curIndex
            if (curSnap <= haloEndSnap):
                subhaloHaloEnd = curHalo
                subhaloHaloEndSnap = curSnap
                subhaloHaloEndIndex = curIndex
            if (curHalo == curHead):
                break
            curHalo = curHead
            curSnap = np.uint64(curHalo / TEMPORALHALOIDVAL)
            curIndex = np.uint64(curHalo % TEMPORALHALOIDVAL - 1)
            curRootTail = halodata[curSnap]['RootTail'][curIndex]
            curHead = halodata[curSnap]['Head'][curIndex]
        # if subhalo exists for longer than halo, take over subhalo descendant line
        # at the point where the halo ceases to exist
        if (haloEndSnap < subhaloEndSnap) :
            if (iverbose > 1):
                print('halo taking over subhalo descendent',haloEnd,subhaloEnd,subhaloHaloEnd)
            halodata[haloEndSnap]['Head'][haloEndIndex] = halodata[subhaloHaloEndSnap]['Head'][subhaloHaloEndIndex]
            halodata[haloEndSnap]['HeadSnap'][haloEndIndex] = halodata[subhaloHaloEndSnap]['HeadSnap'][subhaloHaloEndIndex]
            subhaloHeadFix = halodata[subhaloHaloEndSnap]['Head'][subhaloHaloEndIndex]
            subhaloHeadFixSnap = np.uint64(curHalo / TEMPORALHALOIDVAL)
            subhaloHeadFixIndex = np.uint64(curHalo % TEMPORALHALOIDVAL - 1)
            halodata[subhaloHeadFixSnap]['Tail'][subhaloHeadFixIndex] = haloEnd
            halodata[subhaloHeadFixSnap]['TailSnap'][subhaloHeadFixIndex] = haloEndSnap
            curHalo = branchfixSwapBranch
            curSnap = np.uint64(curHalo / TEMPORALHALOIDVAL)
            curIndex = np.uint64(curHalo % TEMPORALHALOIDVAL - 1)
            curHead = halodata[curSnap]['Head'][curIndex]
            while (curSnap < subhaloHeadFixSnap):
                halodata[curSnap]['RootTail'][curIndex] = branchfixSwapBranch
                halodata[curSnap]['RootTailSnap'][curIndex] = branchfixSnap
                curHalo = curHead
                curSnap = np.uint64(curHalo / TEMPORALHALOIDVAL)
                curIndex = np.uint64(curHalo % TEMPORALHALOIDVAL - 1)
                curRootTail = halodata[curSnap]['RootTail'][curIndex]
                curHead = halodata[curSnap]['Head'][curIndex]
    #if subhalo swap, need to generalise to subhalo merge point later
    else :
        branchfixSubhaloIndex = np.uint64(branchfixSwapBranchSubhalo % TEMPORALHALOIDVAL-1)
        branchfixSubhaloSnap = np.uint64(branchfixSwapBranchSubhalo / TEMPORALHALOIDVAL)
        haloHead = halodata[branchfixSubhaloSnap]['Head'][branchfixSubhaloIndex]
        haloHeadSnap = np.uint64(haloHead / TEMPORALHALOIDVAL)
        haloHeadIndex = np.uint64(haloHead % TEMPORALHALOIDVAL-1)
        haloRootHead = halodata[haloSnap]['RootHead'][haloIndex]


        # currently code checks to see if halo merges immediately with subhalo main
        # branch or merges later as the interpretation of branchfix parameters
        # are interpreted differently. Need to clean up this code and
        # clean up the interface
        # need more useful/informative flag but if Tail is -1
        # then simple immediate halo into subhalo merger, adjust the head using halo head
        if (branchfixSwapBranchTail == -1):
            # have halo point to subhalo descedant and descendant point to halo
            halodata[branchfixSnap]['Head'][branchfixIndex] = haloHead
            halodata[branchfixSnap]['HeadSnap'][branchfixIndex] = haloHeadSnap
            halodata[haloHeadSnap]['Tail'][haloHeadIndex] = branchfixSwapBranch
            halodata[haloHeadSnap]['TailSnap'][haloHeadIndex] = branchfixSnap
            # and alter the main branch line of the subhalo to point to the halo's main branch line's root tail
            curHalo = haloHead
            curSnap = np.uint64(curHalo / TEMPORALHALOIDVAL)
            curIndex = np.uint64(curHalo % TEMPORALHALOIDVAL - 1)
            # curTail = halodata[curSnap]['Tail'][curIndex]
            curRootTail = halodata[curSnap]['RootTail'][curIndex]
            while (curRootTail == haloID):
                if (iverbose > 1):
                    print('subhalo swap fix, moving up branch to adjust the root tails', curHalo,
                          curSnap, halodata[curSnap]['RootTail'][curIndex], branchfixRootTail)

                halodata[curSnap]['RootTail'][curIndex] = branchfixRootTail
                halodata[curSnap]['RootTailSnap'][curIndex] = branchfixRootTailSnap
                if (curHalo == haloRootHead):
                    break
                curHalo = halodata[curSnap]['Head'][curIndex]
                curSnap = np.uint64(curHalo / TEMPORALHALOIDVAL)
                curIndex = np.uint64(curHalo % TEMPORALHALOIDVAL - 1)
                curRootTail = halodata[curSnap]['RootTail'][curIndex]
        # otherwise assume that had to move forward in time and need adjust where we change pointers
        else:
            newHead = branchfixSwapBranchTail
            newSnap = np.uint64(newHead / TEMPORALHALOIDVAL)
            newIndex = np.uint64(newHead % TEMPORALHALOIDVAL - 1)
            # need the swap check code to return more stuff I think
            # have halo point to what should be subhalo first as halo
            halodata[branchfixSnap]['Head'][branchfixIndex] = newHead
            halodata[branchfixSnap]['HeadSnap'][branchfixIndex] = newSnap
            # then have that object point to the halo which should be stored in branchfix
            halodata[newSnap]['Tail'][newIndex] = branchfixSwapBranch
            halodata[newSnap]['TailSnap'][newIndex] = branchfixSnap
            # and alter the main branch line of the subhalo to point to the halo's main branch line's root tail
            curHalo = newHead
            curSnap = np.uint64(curHalo / TEMPORALHALOIDVAL)
            curIndex = np.uint64(curHalo % TEMPORALHALOIDVAL - 1)
            # curTail = halodata[curSnap]['Tail'][curIndex]
            curRootTail = halodata[curSnap]['RootTail'][curIndex]
            while (curRootTail == haloID):
                if (iverbose > 1):
                    print('subhalo swap fix, moving up branch to adjust the root tails', curHalo,
                          curSnap, halodata[curSnap]['RootTail'][curIndex], branchfixRootTail)

                halodata[curSnap]['RootTail'][curIndex] = branchfixRootTail
                halodata[curSnap]['RootTailSnap'][curIndex] = branchfixRootTailSnap
                if (curHalo == haloRootHead):
                    break
                curHalo = halodata[curSnap]['Head'][curIndex]
                curSnap = np.uint64(curHalo / TEMPORALHALOIDVAL)
                curIndex = np.uint64(curHalo % TEMPORALHALOIDVAL - 1)
                curRootTail = halodata[curSnap]['RootTail'][curIndex]
            # and if branch Head indicates trying to match sub to sub, lets adjust that
            if (branchfixSwapBranchHead > -1):
                branchfixSwapBranchHeadIndex = np.uint64(branchfixSwapBranchHead % TEMPORALHALOIDVAL-1)
                branchfixSwapBranchHeadSnap = np.uint64(branchfixSwapBranchHead / TEMPORALHALOIDVAL)

                halodata[branchfixSubhaloSnap]['Head'][branchfixSubhaloIndex] = branchfixSwapBranchHead
                halodata[branchfixSubhaloSnap]['HeadSnap'][branchfixSubhaloIndex] = branchfixSwapBranchHeadSnap
                halodata[branchfixSwapBranchHeadSnap]['Tail'][branchfixSwapBranchHeadIndex] = branchfixSwapBranchSubhalo
                halodata[branchfixSwapBranchHeadSnap]['TailSnap'][branchfixSwapBranchHeadIndex] = branchfixSubhaloSnap
                #and adjust root tails
                curHalo = branchfixSwapBranchHead
                curSnap = np.uint64(curHalo / TEMPORALHALOIDVAL)
                curIndex = np.uint64(curHalo % TEMPORALHALOIDVAL - 1)
                oldRootTail = curRootTail = halodata[curSnap]['RootTail'][curIndex]
                newRootTail = halodata[branchfixSubhaloSnap]['RootTail'][branchfixSubhaloIndex]
                newRootTailSnap = np.uint64(curHalo / TEMPORALHALOIDVAL)
                while (curRootTail == oldRootTail):
                    if (iverbose > 1):
                        print('subhalo swap fix, moving up branch to adjust the root tails', curHalo,
                              curSnap, halodata[curSnap]['RootTail'][curIndex], newRootTail)
                    halodata[curSnap]['RootTail'][curIndex] = newRootTail
                    halodata[curSnap]['RootTailSnap'][curIndex] = newRootTailSnap
                    if (curHalo == haloRootHead):
                        break
                    curHalo = halodata[curSnap]['Head'][curIndex]
                    curSnap = np.uint64(curHalo / TEMPORALHALOIDVAL)
                    curIndex = np.uint64(curHalo % TEMPORALHALOIDVAL - 1)
                    curRootTail = halodata[curSnap]['RootTail'][curIndex]

def FixTruncationBranchSwapsInTreeDescendant(numsnaps, treedata, halodata, numhalos,
                                             npartlim=200, meritlim=0.025, xdifflim=2.0, vdifflim=1.0, nsnapsearch=4,
                                             searchdepth=2, iswaphalosubhaloflag=1,
                                             TEMPORALHALOIDVAL=1000000000000, iverbose=1,
                                             ichecktree = False
                                             ):
    """
    Updates the walkable tree information stored with the halo data
    by using the raw tree produced by TreeFrog to correct any branch swap events leading to truncation
    Requires full roothead/root tail information to correctly fix. Also requires full
    raw tre information with merits and secondary rank descendants.

    Goal is to identify objects with no progenitors that are too large to have no progenitors.
    This should be based on the limit in the halo catalog, typically 10*nhalocataloglim is likely
    an good choice.

    For these objects, the code tries to identify instances of branch swap and truncation events resulting
    from objects being missing from the input halo catalog for several snapshots, specifically using secondary
    descendant information to identify possible connections.

    For objects with no progenitor, it examines all objects that share the same root descendant
    with a snapsearch window in the past, looking for objects that list the halo as a secondary (rank=1)
    descendant with a high enough merit, that is have a fragmentation event pointing an object being
    missing from the input halo catalogue.

    Given the fragementation event, the code searches the objects prior to the merge event that also
    belong to the same root descendant for an object that might match the object in phase-space and hence
    be a possible progenitor. If found, the branches are fixed and also the head, tails (and root head and root tails).

    If no viable progenitor is found, the code can also proceed to adjust the head/tail structure of the tree so that
    a main branch (and halos) are not left with large root tails. If a subhalo is identified as having no progenitor
    but is the main branch of the root head and its host mergers with subhalo branch then alter to host halo defining the main
    branch, and subhalo becoming secondary progenitor living for a short period of time. If object is a halo, then identify subhalo
    that defines the main branch and swap objects.


    """
    start = time.process_time()
    start0 = time.process_time()
    print('Starting to fix branches, examing',np.sum(numhalos),'across',numsnaps)
    SimulationInfo = copy.deepcopy(halodata[0]['SimulationInfo'])
    UnitInfo = copy.deepcopy(halodata[0]['UnitInfo'])
    period = SimulationInfo['Period']
    converttocomove = ['Xc', 'Yc', 'Zc', 'Rmax', 'R_200crit']
    keys = halodata[0].keys()
    for key in converttocomove:
        if key not in keys:
            converttocomove.remove(key)
    # convert positions and sizes to comoving if necesary
    if (UnitInfo['Comoving_or_Physical'] == 0 and SimulationInfo['Cosmological_Sim'] == 1):
        print('Converting to comoving')
        for i in range(numsnaps):
            for key in converttocomove:
                halodata[i][key] /= halodata[i]['SimulationInfo']['ScaleFactor']
        # extracted period from first snap so can use the scale factor stored in simulation info
        period /= SimulationInfo['ScaleFactor']
    print(time.process_time()-start0)
    # store number of fixes
    fixkeylist = ['TotalOutliers', 'HaloOutliers', 'SubOutliers',
                  'AfterFixTotalOutliers', 'AfterFixHaloOutliers', 'AfterFixSubOutliers',
                  'TotalFix', 'MergeFix', 'MergeFixBranchSwap', 'HaloSwapFix', 'SubSwapFix',
                  'NoFixAll', 'NoFixMerge', 'NoFixMergeBranchSwap', 'NoFixHaloSwap', 'NoFixSubSwap',
                  'NoMergeCandiate', 'Spurious']
    nfix = dict()
    for key in fixkeylist:
        nfix[key]= np.zeros(numsnaps)

    start1=time.process_time()

    temparray = {'RootHead':np.array([], dtype=np.int64), 'ID': np.array([], dtype=np.int64), 'npart': np.array([], dtype=np.int32),
                'Descen': np.array([], dtype=np.int64), 'Rank': np.array([], dtype=np.int32), 'Merit': np.array([], np.float32),
                }
    secondaryProgenList = {'RootHead':np.array([], dtype=np.int64), 'ID': np.array([], dtype=np.int64), 'npart': np.array([], dtype=np.int32),
                'Structuretype':np.array([],np.int32),
                'Descen': np.array([], dtype=np.int64), 'Rank': np.array([], dtype=np.int32), 'Merit': np.array([], np.float32),
                }
    num_with_more_descen = 0
    num_secondary_progen = 0
    #limit possible matches to near particle limit of viable no progenitor halos
    npartlim_fragmentation = npartlim_secondary = npartlim/5
    # make flatten array of tree structure within temporal search window to
    # speed up process of searching for related objects
    for isearch in range(numsnaps):
        if (numhalos[isearch] == 0):
            continue
        #store objects that fragment
        for idepth in range(1,searchdepth+1):
            wdata = np.where((treedata[isearch]['Num_descen']>idepth)*(halodata[isearch]['npart']>=npartlim_fragmentation))[0]
            if (wdata.size == 0):
                continue
            num_with_more_descen += wdata.size
            temparray['RootHead'] = np.concatenate([temparray['RootHead'],halodata[isearch]['RootHead'][wdata]])
            temparray['ID'] = np.concatenate([temparray['ID'],halodata[isearch]['ID'][wdata]])
            temparray['npart'] = np.concatenate([temparray['npart'],halodata[isearch]['npart'][wdata]])
            if ('_Offsets' in treedata[isearch].keys()):
                temptemparray = treedata[isearch]['_Offsets'][wdata]+idepth
                temparray['Descen'] = np.concatenate([temparray['Descen'],treedata[isearch]['_Descens'][temptemparray]])
                temparray['Rank'] = np.concatenate([temparray['Rank'],treedata[isearch]['_Ranks'][temptemparray]])
                temparray['Merit'] = np.concatenate([temparray['Merit'],treedata[isearch]['_Merits'][temptemparray]])
            else:
                temptemparray=np.zeros(wdata.size, dtype=np.int64)
                for iw in range(wdata.size):
                    temptemparray[iw]=treedata[isearch]['Descen'][wdata[iw]][idepth]
                temparray['Descen'] = np.concatenate([temparray['Descen'],temptemparray])
                temptemparray=np.zeros(wdata.size, dtype=np.int32)
                for iw in range(wdata.size):
                    temptemparray[iw]=treedata[isearch]['Rank'][wdata[iw]][idepth]
                temparray['Rank'] = np.concatenate([temparray['Rank'],temptemparray])
                temptemparray=np.zeros(wdata.size, dtype=np.float32)
                for iw in range(wdata.size):
                    temptemparray[iw]=treedata[isearch]['Merit'][wdata[iw]][idepth]
                temparray['Merit'] = np.concatenate([temparray['Merit'],temptemparray])
        #store secondary progenitors
        for idepth in range(0,searchdepth):
            wdata = np.where((treedata[isearch]['Num_descen']>idepth)*(halodata[isearch]['npart']>=npartlim_secondary))[0]
            if (wdata.size == 0):
                continue
            if ('_Offsets' in treedata[isearch].keys()):
                temptemparray = treedata[isearch]['_Offsets'][wdata]+idepth
                wdata = wdata[np.where(treedata[isearch]['_Ranks'][temptemparray]>0)]
            else:
                temptemparray=np.zeros(wdata.size, dtype=np.int64)
                icount = 0
                for iw in range(wdata.size):
                    if (treedata[isearch]['Rank'][wdata[iw]][idepth]>0):
                        temptemparray[icount] = wdata[iw]
                        icount += 1
                wdata = temptemparray[:icount]
            num_secondary_progen += wdata.size
            secondaryProgenList['RootHead'] = np.concatenate([secondaryProgenList['RootHead'],halodata[isearch]['RootHead'][wdata]])
            secondaryProgenList['ID'] = np.concatenate([secondaryProgenList['ID'],halodata[isearch]['ID'][wdata]])
            secondaryProgenList['npart'] = np.concatenate([secondaryProgenList['npart'],halodata[isearch]['npart'][wdata]])
            secondaryProgenList['Structuretype'] = np.concatenate([secondaryProgenList['Structuretype'],halodata[isearch]['Structuretype'][wdata]])
            if ('_Offsets' in treedata[isearch].keys()):
                temptemparray = treedata[isearch]['_Offsets'][wdata]+idepth
                secondaryProgenList['Descen'] = np.concatenate([secondaryProgenList['Descen'],treedata[isearch]['_Descens'][temptemparray]])
                secondaryProgenList['Rank'] = np.concatenate([secondaryProgenList['Rank'],treedata[isearch]['_Ranks'][temptemparray]])
                secondaryProgenList['Merit'] = np.concatenate([secondaryProgenList['Merit'],treedata[isearch]['_Merits'][temptemparray]])
            else:
                temptemparray=np.zeros(wdata.size, dtype=np.int64)
                for iw in range(wdata.size):
                    temptemparray[iw]=treedata[isearch]['Descen'][wdata[iw]][idepth]
                secondaryProgenList['Descen'] = np.concatenate([secondaryProgenList['Descen'],temptemparray])
                temptemparray=np.zeros(wdata.size, dtype=np.int32)
                for iw in range(wdata.size):
                    temptemparray[iw]=treedata[isearch]['Rank'][wdata[iw]][idepth]
                secondaryProgenList['Rank'] = np.concatenate([secondaryProgenList['Rank'],temptemparray])
                temptemparray=np.zeros(wdata.size, dtype=np.float32)
                for iw in range(wdata.size):
                    temptemparray[iw]=treedata[isearch]['Merit'][wdata[iw]][idepth]
                secondaryProgenList['Merit'] = np.concatenate([secondaryProgenList['Merit'],temptemparray])
    print('Finished building temporary array for quick search containing ',num_with_more_descen)
    print('Finished building temporary secondary progenitor array for quick search containing ',num_secondary_progen)
    print('in',time.process_time()-start1)

    #find all possible matches to objects with no primary progenitor
    start1 = time.process_time()
    noprogID = np.array([],dtype=np.int64)
    noprogRootHead = np.array([],dtype=np.int64)
    #noprognpart = np.array([],dtype=np.int64)
    for i in range(numsnaps):
        noprog = np.where((halodata[i]['Tail'] == halodata[i]['ID'])*
            (halodata[i]['npart'] >= npartlim)
            )[0]
        nfix['TotalOutliers'][i] = noprog.size
        nfix['HaloOutliers'][i] = np.where(halodata[i]['hostHaloID'][noprog] == -1)[0].size
        nfix['SubOutliers'][i] = nfix['TotalOutliers'][i] - nfix['HaloOutliers'][i]
        if (noprog.size >0):
            noprogID = np.concatenate([noprogID, np.array(halodata[i]['ID'][noprog], dtype=np.int64)])
            noprogRootHead = np.concatenate([noprogRootHead, np.array(halodata[i]['RootHead'][noprog], dtype=np.int64)])
            #noprogRootHead = np.concatenate([noprognpart, np.array(halodata[i]['npart'][noprog], dtype=np.int64)])
    mergeCheck = (np.in1d(temparray['RootHead'],noprogRootHead) *
            np.in1d(temparray['Descen'], noprogID) *
            (temparray['Merit'] >= meritlim) *
            (temparray['Rank'] >= 0 )
            )
    #store the best match based on merit
    mergeListDict = None
    mergeTrue = np.where(mergeCheck)[0]
    tempDescen = np.array([], np.int64)
    tempID = np.array([], np.int64)
    mergeSize = 0
    tempMerit = np.array([], np.float32)
    if (mergeTrue.size > 0):
        #sort by merit
        tempMerit = np.array(temparray['Merit'][mergeTrue], dtype=np.float32)
        sortMerit = np.argsort(tempMerit)[::-1]
        tempDescen = np.array(temparray['Descen'][mergeTrue][sortMerit], dtype=np.int64)
        tempID = np.array(temparray['ID'][mergeTrue][sortMerit], dtype=np.int64)
        tempMerit = tempMerit[sortMerit]
        sortMerit = None
        #get unique highest merit match for each object
        val, tempDescenUnique = np.unique(tempDescen, return_index=True)
        tempDescen = tempDescen[tempDescenUnique]
        tempID = tempID[tempDescenUnique]
        tempMerit = tempMerit[tempDescenUnique]
        mergeSize = tempDescen.size
        #find all those with no match
        wdata = np.where(np.in1d(noprogID, tempDescen, invert=True))[0]
        if (wdata.size > 0):
            tempDescen = np.concatenate([tempDescen,noprogID[wdata]])
            tempID = np.concatenate([tempID,np.int64(-1*np.ones(wdata.size))])
            tempMerit = np.concatenate([tempMerit,-1.0*np.ones(wdata.size)])
    else:
        tempDescen = noprogID
        tempID = np.int64(-1*np.ones(noprogID.size))
        tempMerit = -1.0*np.ones(noprogID.size)
    #store in dictionary
    mergeListDict = dict(zip(tempDescen,tempID))
    mergeListDictMerit = dict(zip(tempDescen,tempMerit))
    tempDescen = tempID = tempMerit = None

    print('Finished determining if objects have possible merge origins.\n'
          'Number without progenitors:',noprogID.size,
          'Number with a possible merger candidate', mergeSize)
    print('Finished initalization in ',time.process_time()-start1)
    print('Now processing ... ')

    # have object with no progenitor
    for haloID in noprogID:
        # have object with no progenitor
        haloSnap = np.uint64(haloID/TEMPORALHALOIDVAL)
        haloIndex = np.uint64(haloID % TEMPORALHALOIDVAL-1)
        haloRootHeadID = halodata[haloSnap]['RootHead'][haloIndex]
        haloHeadID = halodata[haloSnap]['Head'][haloIndex]
        haloHeadSnap = np.uint64(haloHeadID/TEMPORALHALOIDVAL)
        haloHeadIndex = np.uint64(haloHeadID % TEMPORALHALOIDVAL-1)
        mergeHalo = mergeListDict[haloID]
        haloHost = halodata[haloSnap]['hostHaloID'][haloIndex]

        if (haloHost != -1):
            # if object doesn't have a main branch that persists for at least 1 snapshot, skip
            if (mergeHalo == -1 and halodata[haloHeadSnap]['Tail'][haloHeadIndex] != haloID):
                nfix['Spurious'][haloSnap] += 1
                continue

        branchfixMerge = branchfixMergeSwapBranch = -1
        branchfixSwapHaloOrSubhalo = branchfixSwapHaloOrSubhaloSubhaloPoint = branchfixSwapHaloOrSubhaloTail = branchfixSwapHaloOrSubhaloHead = -1

        if (iverbose > 0):
            mainbranchlength = sublifelength = 0
            curHalo = haloID
            curSnap = np.uint64(curHalo / TEMPORALHALOIDVAL)
            curIndex = np.uint64(curHalo % TEMPORALHALOIDVAL - 1)
            curRootTail = halodata[curSnap]['RootTail'][curIndex]
            curRootHead = halodata[curSnap]['RootHead'][curIndex]
            curHead = halodata[curSnap]['Head'][curIndex]
            print(haloID,curRootTail,curRootHead, curHead, '----')
            while (curRootTail == haloID):
                mainbranchlength += 1
                sublifelength += (halodata[curSnap]['hostHaloID'][curIndex] != -1)
                if (curHalo == curRootHead):
                    break
                curHalo = curHead
                curSnap = np.uint64(curHalo / TEMPORALHALOIDVAL)
                curIndex = np.uint64(curHalo % TEMPORALHALOIDVAL - 1)
                curRootTail = halodata[curSnap]['RootTail'][curIndex]
                curHead = halodata[curSnap]['Head'][curIndex]
            print('halo with no progenitor', haloID,
                  'npart=', halodata[haloSnap]['npart'][haloIndex],
                  'Stype=', halodata[haloSnap]['Structuretype'][haloIndex],
                  'branchlength',mainbranchlength,
                  'host=', haloHost,
                  'head=', halodata[haloSnap]['Head'][haloIndex],
                  'head_tail=',halodata[np.uint32(halodata[haloSnap]['Head'][haloIndex]/TEMPORALHALOIDVAL)]['Tail'][np.uint64(halodata[haloSnap]['Head'][haloIndex] % TEMPORALHALOIDVAL-1)],
                  'mergeHalo= ',mergeHalo)

        # if no merge halo candidate found then cannot fully correct tree by assigning
        # object a new progenitor
        # instead can see if object's main branch can be reassigned to another halo
        # in order to make main branches smooth
        if (mergeHalo == -1 and iswaphalosubhaloflag == 1 and haloSnap <numsnaps-1):
            nfix['NoMergeCandiate'][haloSnap] += 1
            branchfixSwapHaloOrSubhalo, branchfixSwapHaloOrSubhaloSubhaloPoint, branchfixSwapHaloOrSubhaloHead, branchfixSwapHaloOrSubhaloTail = FixBranchHaloSubhaloSwapBranch(numsnaps, treedata, halodata, numhalos,
                        period,
                        npartlim, meritlim, xdifflim, vdifflim, nsnapsearch,
                        TEMPORALHALOIDVAL, iverbose,
                        haloID, haloSnap, haloIndex, haloRootHeadID, mergeHalo,
                        secondaryProgenList
                        )
            if (branchfixSwapHaloOrSubhalo > -1):
                nfix['TotalFix'][haloSnap] += 1
                if (halodata[haloSnap]['hostHaloID'][haloIndex] == -1):
                    nfix['HaloSwapFix'][haloSnap] += 1
                else :
                    nfix['SubSwapFix'][haloSnap] += 1
                FixBranchHaloSubhaloSwapBranchAdjustTree(numsnaps, treedata, halodata, numhalos,
                        nsnapsearch,
                        TEMPORALHALOIDVAL, iverbose,
                        haloID, haloSnap, haloIndex, haloRootHeadID,
                        branchfixSwapHaloOrSubhalo, branchfixSwapHaloOrSubhaloSubhaloPoint,
                        branchfixSwapHaloOrSubhaloHead, branchfixSwapHaloOrSubhaloTail
                        )
                continue
            elif (branchfixSwapHaloOrSubhalo == -2):
                if (halodata[haloSnap]['hostHaloID'][haloIndex] == -1):
                    nfix['NoFixHaloSwap'][haloSnap] += 1
                else :
                    nfix['NoFixSubSwap'][haloSnap] += 1
                    nfix['NoFixAll'][haloSnap] +=1
                continue
            else :
                nfix['NoFixAll'][haloSnap] +=1
                continue
        elif (mergeHalo == -1):
            continue

        mergeSnap = np.uint64(mergeHalo / TEMPORALHALOIDVAL)
        mergeIndex = np.uint64(mergeHalo % TEMPORALHALOIDVAL-1)
        if (iverbose > 1):
            print(haloID, 'may have found merge halo', mergeHalo,
                  'merge halo descendants', treedata[mergeSnap]['Descen'][mergeIndex],
                  'merge halo merits', treedata[mergeSnap]['Merit'][mergeIndex],
                  'merge halo ranks', treedata[mergeSnap]['Rank'][mergeIndex])
        # have candidate object that points to both object with no progenitor and object with progenitor
        # store post merge main branch
        postmergeHalo = halodata[mergeSnap]['Head'][mergeIndex]
        postmergeSnap = np.uint64(postmergeHalo/TEMPORALHALOIDVAL)
        postmergeIndex = np.uint64(postmergeHalo % TEMPORALHALOIDVAL-1)
        # note that if post
        # store the progenitor of the halo where likely something has merged
        premergeHalo = halodata[mergeSnap]['Tail'][mergeIndex]
        premergeSnap = np.uint64(premergeHalo/TEMPORALHALOIDVAL)
        premergeIndex = np.uint64(premergeHalo % TEMPORALHALOIDVAL-1)
        # get the merge object's pre/cur/post host halos
        mergeHost = halodata[mergeSnap]['hostHaloID'][mergeIndex]
        if (mergeHost == -1):
            mergeHost = halodata[mergeSnap]['ID'][mergeIndex]
        postmergeHost = halodata[postmergeSnap]['hostHaloID'][postmergeIndex]
        if (postmergeHost == -1):
            postmergeHost = halodata[postmergeSnap]['ID'][postmergeIndex]
        premergeHost = halodata[premergeSnap]['hostHaloID'][premergeIndex]
        if (premergeHost == -1):
            premergeHost = halodata[premergeSnap]['ID'][premergeIndex]

        branchfixMerge, branchfixMergeSwapBranch = FixBranchMergePhaseSearch(numsnaps, treedata, halodata, numhalos,
                            period,
                            npartlim, meritlim, xdifflim, vdifflim, nsnapsearch,
                            TEMPORALHALOIDVAL, iverbose,
                            haloID, haloSnap, haloIndex, haloRootHeadID,
                            mergeHalo, mergeSnap, mergeIndex,
                            premergeHalo, premergeSnap, premergeIndex,
                            postmergeHalo,postmergeSnap, postmergeIndex,
                            secondaryProgenList
                            )
        if (branchfixMerge < 0):
            nfix['NoFixMerge'][haloSnap] += 1
        if (branchfixMergeSwapBranch < 0):
            nfix['NoFixMergeBranchSwap'][haloSnap] += 1
        if (branchfixMerge < 0 and branchfixMergeSwapBranch < 0 and iswaphalosubhaloflag == 0):
            continue
        if (branchfixMerge < 0 and branchfixMergeSwapBranch < 0):
            branchfixSwapHaloOrSubhalo, branchfixSwapHaloOrSubhaloSubhaloPoint, branchfixSwapHaloOrSubhaloHead, branchfixSwapHaloOrSubhaloTail = FixBranchHaloSubhaloSwapBranch(numsnaps, treedata, halodata, numhalos,
                        period,
                        npartlim, meritlim, xdifflim, vdifflim, nsnapsearch,
                        TEMPORALHALOIDVAL, iverbose,
                        haloID, haloSnap, haloIndex, haloRootHeadID, mergeHalo,
                        secondaryProgenList
                        )
        if (branchfixMerge < 0 and branchfixMergeSwapBranch < 0 and
            branchfixMergeSwapBranch < 0 and
            branchfixSwapHaloOrSubhalo < 0):
            if (iverbose > 0):
                print('halo ', haloID, 'cannot be FIXED!')
            nfix['NoFixAll'][haloSnap] += 1
            if (branchfixSwapHaloOrSubhalo == -2):
                if (halodata[haloSnap]['hostHaloID'][haloIndex] == -1):
                    nfix['NoFixHaloSwap'][haloSnap] += 1
                else :
                    nfix['NoFixSubSwap'][haloSnap] += 1
            continue

        # There is a fix so lets adjust the tree
        nfix['TotalFix'][haloSnap] += 1
        if (branchfixMerge > -1):
            nfix['MergeFix'][haloSnap] += 1
            FixBranchPhaseMergeAdjustTree(numsnaps, treedata, halodata, numhalos,
                TEMPORALHALOIDVAL, iverbose,
                haloID, haloSnap, haloIndex,
                mergeHalo, mergeSnap, mergeIndex,
                premergeHalo, premergeSnap, premergeIndex,
                postmergeHalo,postmergeSnap, postmergeIndex,
                branchfixMerge
            )
        elif (branchfixMergeSwapBranch > -1):
            nfix['MergeFixBranchSwap'][haloSnap] += 1
            FixBranchPhaseBranchSwapAdjustTree(numsnaps, treedata, halodata, numhalos,
                TEMPORALHALOIDVAL, iverbose,
                haloID, haloSnap, haloIndex,
                mergeHalo, mergeSnap, mergeIndex,
                premergeHalo, premergeSnap, premergeIndex,
                postmergeHalo, postmergeSnap, postmergeIndex,
                branchfixMergeSwapBranch
            )
        elif (iswaphalosubhaloflag == 1):
            if (branchfixSwapHaloOrSubhalo > -1):
                if (halodata[haloSnap]['hostHaloID'][haloIndex] == -1):
                    nfix['HaloSwapFix'][haloSnap] += 1
                else :
                    nfix['SubSwapFix'][haloSnap] += 1
                FixBranchHaloSubhaloSwapBranchAdjustTree(numsnaps, treedata, halodata, numhalos,
                        nsnapsearch,
                        TEMPORALHALOIDVAL, iverbose,
                        haloID, haloSnap, haloIndex, haloRootHeadID,
                        branchfixSwapHaloOrSubhalo, branchfixSwapHaloOrSubhaloSubhaloPoint,
                        branchfixSwapHaloOrSubhaloHead, branchfixSwapHaloOrSubhaloTail
                        )
            elif (branchfixSwapHaloOrSubhalo == -2):
                if (halodata[haloSnap]['hostHaloID'][haloIndex] == -1):
                    nfix['NoFixHalo'][haloSnap] += 1
                else :
                    nfix['NoFixSub'][haloSnap] += 1
                nfix['NoFixAll'][haloSnap] +=1
                continue
            else :
                nfix['NoFixAll'][haloSnap] +=1
                continue

    #do last snapshot, fixing halos only.


    # if checking tree, make sure root heads and root tails match when head and tails indicate they should
    if (ichecktree):
        irebuildrootheadtail = False
        print('Checking tree root/tail structure ...')
        for i in range(numsnaps):
            ntailcheck = np.where((halodata[i]['Tail'] == halodata[i]['ID'])*
                (halodata[i]['RootTail'] != halodata[i]['Tail']))[0].size
            nheadcheck = np.where((halodata[i]['Head'] == halodata[i]['ID'])*
                (halodata[i]['RootHead'] != halodata[i]['Head']))[0].size
            if (ntailcheck > 0):
                print(i, 'Issue with RootTail! Might need to rebuild Root Tails or check input tree', ntailcheck)
                irebuildrootheadtail = True
            if (nheadcheck > 0):
                print(i, 'Issue with RootHead! Might need to rebuild Root Head or check input tree', nheadcheck)
                irebuildrootheadtail = True
        if (irebuildrootheadtail):
            for i in range(numsnaps):
                roottails = np.where((halodata[i]['Tail'] == halodata[i]['ID']))
                # now should I actually go about and set the root tails of stuff?
        print('Done')


    for i in range(numsnaps):
        noprog = np.where((halodata[i]['Tail'] == halodata[i]['ID'])*(
            halodata[i]['npart'] >= npartlim)*(halodata[i]['Head'] != halodata[i]['ID']))[0]
        nfix['AfterFixTotalOutliers'][i] = noprog.size
        nfix['AfterFixHaloOutliers'][i] = np.where(halodata[i]['hostHaloID'][noprog] == -1)[0].size
        nfix['AfterFixSubOutliers'][i] = nfix['AfterFixTotalOutliers'][i] - nfix['AfterFixHaloOutliers'][i]
    print('Done fixing branches', time.process_time()-start)
    print('For', np.sum(numhalos), 'across cosmic time')
    print('Corrections are:')
    for key in fixkeylist:
        print(key, np.sum(nfix[key]))
    #if (iverbose > 1):
    #    print('With snapshot break down of ')
    #    for i in range(numsnaps):
    #        print('snap', i, 'with', numhalos[i], 'Fixes:')
    #        print([[key, nfix[key][i]] for key in fixkeylist])
    # convert back to physical coordinates if necessary
    if (UnitInfo['Comoving_or_Physical'] == 0 and SimulationInfo['Cosmological_Sim'] == 1):
        converttocomove = ['Xc', 'Yc', 'Zc', 'Rmax', 'R_200crit']
        keys=halodata[0].keys()
        for i in range(numsnaps):
            for key in converttocomove:
                if key not in keys: continue
                halodata[i][key] *= halodata[i]['SimulationInfo']['ScaleFactor']
        # extracted period from first snap so can use the scale factor stored in simulation info
        period /= SimulationInfo['ScaleFactor']


def CleanSecondaryProgenitorsFromNoPrimaryProgenObjectsTreeDescendant(numsnaps, treedata, halodata, numhalos,
                                             npartlim=200,
                                             TEMPORALHALOIDVAL=1000000000000, iverbose=1):
    """
    Identifies all secondary progenitors and if they point to an object lacking
    a primary progenior, they either point to
    1) the object's host (if object is subhalo and host has a progenitor and root head stays the same)
    2) Objct's descendant (moving up till descendant is found having a primary progenitor)

    This code clean's up tree for SAMs that simple walk trees forward without checking
    if moving foward is main branch or not.
    """

    start = time.process_time()
    start0 = time.process_time()
    print('Starting secondary progenitor clean-up examing',np.sum(numhalos),'across',numsnaps)
    # store number of fixes
    fixkeylist = ['NoPrimary', 'SecondarytoNoPrimary', 'HostHaloFix', 'DescendantFix', 'NoFix']
    nfix = dict()
    for key in fixkeylist:
        nfix[key]= np.zeros(numsnaps)
    secondaryProgenList = {'RootHead':np.array([], dtype=np.int64), 'Head':np.array([], dtype=np.int64),
                            'ID': np.array([], dtype=np.int64),
                            }
    num_secondary_progen = 0
    start1=time.process_time()
    # make flatten array of tree structure within temporal search window to
    # speed up process of searching for related objects
    for isearch in range(numsnaps):
        if (numhalos[isearch] == 0):
            continue
        wdata = np.where((treedata[isearch]['Num_descen']>0))[0]
        if (wdata.size == 0):
            continue
        if ('_Offsets' in treedata[isearch].keys()):
            temptemparray = treedata[isearch]['_Offsets'][wdata]
            wdata = wdata[np.where(treedata[isearch]['_Ranks'][temptemparray]>0)]
        else:
            temptemparray=np.zeros(wdata.size, dtype=np.int64)
            icount = 0
            for iw in range(wdata.size):
                if (treedata[isearch]['Rank'][wdata[iw]][0]>0):
                    temptemparray[icount] = wdata[iw]
                    icount += 1
            wdata = temptemparray[:icount]
        num_secondary_progen += wdata.size
        secondaryProgenList['RootHead'] = np.concatenate([secondaryProgenList['RootHead'],halodata[isearch]['RootHead'][wdata]])
        secondaryProgenList['Head'] = np.concatenate([secondaryProgenList['Head'],halodata[isearch]['Head'][wdata]])
        secondaryProgenList['ID'] = np.concatenate([secondaryProgenList['ID'],halodata[isearch]['ID'][wdata]])
    print('Finished building temporary secondary progenitor array for quick search containing ',num_secondary_progen)
    print('in',time.process_time()-start1)

    #find all objects with no primary progenitor
    start1 = time.process_time()
    for i in range(numsnaps-1):
        if (numhalos[i] == 0):
            continue
        # get all objects with no primary progenitor that are not their own root head
        noprog = np.where((halodata[i]['Tail'] == halodata[i]['ID'])*
                 (halodata[i]['Head'] != halodata[i]['ID'])*
                 (halodata[i]['npart'] >= npartlim)
                 )[0]
        if (noprog.size == 0):
            continue
        noprogID = np.array(halodata[i]['ID'][noprog], dtype=np.int64)
        nfix['NoPrimary'][i] = noprogID.size
        if (iverbose > 0):
            print('Snapshot ', i, 'with no progenitor ',noprog.size)
        #get all secondary progenitors to these objects
        secondaryCheck = np.in1d(secondaryProgenList['Head'], noprogID)
        secondaryActive = np.where(secondaryCheck)[0]
        if (secondaryActive.size == 0):
            continue
        if (iverbose > 0):
            print('Snapshot ', i, 'active secondaries',secondaryActive.size)
        # for each secondary progenitor adjust its descendant accordingly
        nfix['SecondarytoNoPrimary'][i] = secondaryActive.size
        for j in range(secondaryActive.size):
            sID, sHead = secondaryProgenList['ID'][secondaryActive[j]], secondaryProgenList['Head'][secondaryActive[j]]
            sSnap = np.uint64(sID / TEMPORALHALOIDVAL)
            sIndex = np.uint64(sID % TEMPORALHALOIDVAL - 1)
            sHeadSnap = np.uint64(sHead / TEMPORALHALOIDVAL)
            sHeadIndex = np.uint64(sHead % TEMPORALHALOIDVAL - 1)
            sHeadHost = halodata[sHeadSnap]['hostHaloID'][sHeadIndex]
            sHeadHostRootHead = sHeadHostSnap = sHeadHostIndex = -1
            if (sHeadHost != -1):
                sHeadHostSnap = np.uint64(sHeadHost / TEMPORALHALOIDVAL)
                sHeadHostIndex = np.uint64(sHeadHost % TEMPORALHALOIDVAL - 1)
                sHeadHostRootHead = halodata[sHeadSnap]['RootHead'][sHeadHostIndex]

            #if subhalo has host with a history and the root head remains unchanged, point to host
            if (sHeadHost != -1 and secondaryProgenList['RootHead'][secondaryActive[j]] == sHeadHostRootHead):
                if (halodata[sHeadSnap]['Tail'][sHeadHostIndex] != halodata[sHeadSnap]['ID'][sHeadHostIndex]):
                    halodata[sSnap]['Head'][sIndex] = halodata[sHeadSnap]['ID'][sHeadHostIndex]
                    halodata[sSnap]['RootHead'][sIndex] = halodata[sHeadSnap]['RootHead'][sHeadHostIndex]
                    halodata[sSnap]['RootHeadSnap'][sIndex] = halodata[sHeadSnap]['RootHeadSnap'][sHeadHostIndex]
                    nfix['HostHaloFix'][i] += 1
                    continue
            # otherwise point to subhalo descendant (so long as primary descendant)
            # find first primary descedant along this branch
            curHalo = halodata[sHeadSnap]['ID'][sHeadIndex]
            curSnap = np.uint64(curHalo / TEMPORALHALOIDVAL)
            curIndex = np.uint64(curHalo % TEMPORALHALOIDVAL - 1)
            curHead = halodata[curSnap]['Head'][curIndex]
            curHeadSnap = np.uint64(curHead / TEMPORALHALOIDVAL)
            curHeadIndex = np.uint64(curHead % TEMPORALHALOIDVAL - 1)
            curHeadTail = halodata[curHeadSnap]['Tail'][curHeadIndex]
            curRootHead = halodata[curSnap]['RootHead'][curIndex]

            while (curHeadTail == curHead and curHalo != curRootHead):
                curHalo = curHead
                curSnap = np.uint64(curHalo / TEMPORALHALOIDVAL)
                curIndex = np.uint64(curHalo % TEMPORALHALOIDVAL - 1)
                curHead = halodata[curSnap]['Head'][curIndex]
                curHeadSnap = np.uint64(curHead / TEMPORALHALOIDVAL)
                curHeadIndex = np.uint64(curHead % TEMPORALHALOIDVAL - 1)
                curHeadTail = halodata[curHeadSnap]['Tail'][curHeadIndex]

            halodata[sSnap]['Head'][sIndex] = curHead
            halodata[sSnap]['HeadSnap'][sIndex] = curHeadSnap
            nfix['DescendantFix'][i] += 1

    print('Done adjusting secondaries ', time.process_time()-start1)
    print('For', np.sum(numhalos), 'across cosmic time')
    print('Corrections are:')
    for key in fixkeylist:
        print(key, np.sum(nfix[key]))

"""
    HDF5 Tools
"""

def HDF5WriteDataset(hdf5grp, key, data,
    icompress = False, compress_key = 'gzip', compress_level = 6):
    if (icompress):
        return hdf5grp.create_dataset(key, data=data,
            compression=compress_key, compression_opts=compress_level)
    else:
        return hdf5grp.create_dataset(key, data=data)

"""
    Conversion Tools
"""


def ConvertASCIIPropertyFileToHDF(basefilename, iseparatesubfiles=0, iverbose=0):
    """
    Reads an ASCII file and converts it to the HDF format for VELOCIraptor properties files

    """
    inompi = True
    if (iverbose):
        print("reading properties file and converting to hdf",
              basefilename, os.path.isfile(basefilename))
    filename = basefilename+".properties"
    # load header
    if (os.path.isfile(basefilename) == True):
        numfiles = 0
    else:
        filename = basefilename+".properties"+".0"
        inompi = False
        if (os.path.isfile(filename) == False):
            print("file not found")
            return []
    byteoffset = 0
    # load ascii file
    halofile = open(filename, 'r')
    # read header information
    [filenum, numfiles] = halofile.readline().split()
    filenum = int(filenum)
    numfiles = int(numfiles)
    [numhalos, numtothalos] = halofile.readline().split()
    numhalos = np.uint64(numhalos)
    numtothalos = np.uint64(numtothalos)
    names = ((halofile.readline())).split()
    # remove the brackets in ascii file names
    fieldnames = [fieldname.split("(")[0] for fieldname in names]
    halofile.close()

    for ifile in range(numfiles):
        if (inompi == True):
            filename = basefilename+".properties"
            hdffilename = basefilename+".hdf.properties"
        else:
            filename = basefilename+".properties"+"."+str(ifile)
            hdffilename = basefilename+".hdf.properties"+"."+str(ifile)
        if (iverbose):
            print("reading ", filename)
        halofile = open(filename, 'r')
        hdffile = h5py.File(hdffilename, 'w')
        [filenum, numfiles] = halofile.readline().split()
        [numhalos, numtothalos] = halofile.readline().split()
        filenum = int(filenum)
        numfiles = int(numfiles)
        numhalos = np.uint64(numhalos)
        numtothalos = np.uint64(numtothalos)
        # write header info
        hdffile.create_dataset("File_id", data=np.array([filenum]))
        hdffile.create_dataset("Num_of_files", data=np.array([numfiles]))
        hdffile.create_dataset("Num_of_groups", data=np.array([numhalos]))
        hdffile.create_dataset("Total_num_of_groups",
                               data=np.array([numtothalos]))
        halofile.close()
        if (numhalos > 0):
            htemp = np.loadtxt(filename, skiprows=3).transpose()
        else:
            htemp = [[]for ikeys in range(len(fieldnames))]
        for ikeys in range(len(fieldnames)):
            if (fieldnames[ikeys] == "ID"):
                hdffile.create_dataset(fieldnames[ikeys], data=np.array(
                    htemp[ikeys], dtype=np.uint64))
            elif (fieldnames[ikeys] == "ID_mbp"):
                hdffile.create_dataset(fieldnames[ikeys], data=np.array(
                    htemp[ikeys], dtype=np.int64))
            elif (fieldnames[ikeys] == "hostHaloID"):
                hdffile.create_dataset(fieldnames[ikeys], data=np.array(
                    htemp[ikeys], dtype=np.int64))
            elif fieldnames[ikeys] in ["numSubStruct", "npart", "n_gas", "n_star"]:
                hdffile.create_dataset(fieldnames[ikeys], data=np.array(
                    htemp[ikeys], dtype=np.uint64))
            else:
                hdffile.create_dataset(fieldnames[ikeys], data=np.array(
                    htemp[ikeys], dtype=np.float64))

        hdffile.close()
    # if subhalos are written in separate files, then read them too
    if (iseparatesubfiles == 1):
        for ifile in range(numfiles):
            if (inompi == True):
                filename = basefilename+".sublevels"+".properties"
                hdffilename = basefilename+".hdf"+".sublevels"+".properties"
            else:
                filename = basefilename+".sublevels" + \
                    ".properties"+"."+str(ifile)
                hdffilename = basefilename+".hdf" + \
                    ".sublevels"+".properties"+"."+str(ifile)
            if (iverbose):
                print("reading ", filename)
            halofile = open(filename, 'r')
            hdffile = h5py.File(hdffilename, 'w')
            [filenum, numfiles] = halofile.readline().split()
            [numhalos, numtothalos] = halofile.readline().split()
            filenum = int(filenum)
            numfiles = int(numfiles)
            numhalos = np.uint64(numhalos)
            numtothalos = np.uint64(numtothalos)
            # write header info
            hdffile.create_dataset("File_id", data=np.array([filenum]))
            hdffile.create_dataset("Num_of_files", data=np.array([numfiles]))
            hdffile.create_dataset("Num_of_groups", data=np.array([numhalos]))
            hdffile.create_dataset("Total_num_of_groups",
                                   data=np.array([numtothalos]))
            halofile.close()
            if (numhalos > 0):
                htemp = np.loadtxt(filename, skiprows=3).transpose()
            else:
                htemp = [[]for ikeys in range(len(fieldnames))]
            for ikeys in range(len(fieldnames)):
                if (fieldnames[ikeys] == "ID"):
                    hdffile.create_dataset(fieldnames[ikeys], data=np.array(
                        htemp[ikeys], dtype=np.uint64))
                elif (fieldnames[ikeys] == "ID_mbp"):
                    hdffile.create_dataset(fieldnames[ikeys], data=np.array(
                        htemp[ikeys], dtype=np.int64))
                elif (fieldnames[ikeys] == "hostHaloID"):
                    hdffile.create_dataset(fieldnames[ikeys], data=np.array(
                        htemp[ikeys], dtype=np.int64))
                elif fieldnames[ikeys] in ["numSubStruct", "npart", "n_gas", "n_star"]:
                    hdffile.create_dataset(fieldnames[ikeys], data=np.array(
                        htemp[ikeys], dtype=np.uint64))
                else:
                    hdffile.create_dataset(fieldnames[ikeys], data=np.array(
                        htemp[ikeys], dtype=np.float64))
            hdffile.close()


def ConvertASCIICatalogGroupsFileToHDF(basefilename, iseparatesubfiles=0, iverbose=0):
    """
    Reads an ASCII file and converts it to the HDF format for VELOCIraptor files

    """
    inompi = True
    if (iverbose):
        print("reading properties file and converting to hdf",
              basefilename, os.path.isfile(basefilename))
    filename = basefilename+".catalog_groups"
    # load header
    if (os.path.isfile(basefilename) == True):
        numfiles = 0
    else:
        filename = basefilename+".catalog_groups"+".0"
        inompi = False
        if (os.path.isfile(filename) == False):
            print("file not found")
            return []
    byteoffset = 0
    # load ascii file
    halofile = open(filename, 'r')
    # read header information
    [filenum, numfiles] = halofile.readline().split()
    filenum = int(filenum)
    numfiles = int(numfiles)
    [numhalos, numtothalos] = halofile.readline().split()
    numhalos = np.uint64(numhalos)
    numtothalos = np.uint64(numtothalos)
    halofile.close()

    fieldnames = ["Group_Size", "Offset", "Offset_unbound",
                  "Number_of_substructures_in_halo", "Parent_halo_ID"]
    fieldtype = [np.uint32, np.uint64, np.uint64, np.uint32, np.int64]

    for ifile in range(numfiles):
        if (inompi == True):
            filename = basefilename+".catalog_groups"
            hdffilename = basefilename+".hdf.catalog_groups"
        else:
            filename = basefilename+".catalog_groups"+"."+str(ifile)
            hdffilename = basefilename+".hdf.catalog_groups"+"."+str(ifile)
        if (iverbose):
            print("reading ", filename)
        halofile = open(filename, 'r')
        hdffile = h5py.File(hdffilename, 'w')
        [filenum, numfiles] = halofile.readline().split()
        [numhalos, numtothalos] = halofile.readline().split()
        filenum = int(filenum)
        numfiles = int(numfiles)
        numhalos = np.uint64(numhalos)
        numtothalos = np.uint64(numtothalos)
        # write header info
        hdffile.create_dataset("File_id", data=np.array([filenum]))
        hdffile.create_dataset("Num_of_files", data=np.array([numfiles]))
        hdffile.create_dataset("Num_of_groups", data=np.array([numhalos]))
        hdffile.create_dataset("Total_num_of_groups",
                               data=np.array([numtothalos]))
        halofile.close()
        if (numhalos > 0):
            # will look like one dimensional array of values split into
            # "Group_Size"
            # "Offset"
            # "Offset_unbound"
            # "Number_of_substructures_in_halo"
            # "Parent_halo_ID"
            # each of size numhalos
            cattemp = np.loadtxt(filename, skiprows=2).transpose()
            for ikeys in range(len(fieldnames)):
                hdffile.create_dataset(fieldnames[ikeys], data=np.array(
                    cattemp[ikeys*numhalos:(ikeys+1)*numhalos], dtype=fieldtype[ikeys]))
        else:
            cattemp = []
            for ikeys in range(len(fieldnames)):
                hdffile.create_dataset(
                    fieldnames[ikeys], data=np.array([], dtype=fieldtype[ikeys]))
        hdffile.close()
    # if subhalos are written in separate files, then read them too
    if (iseparatesubfiles == 1):
        for ifile in range(numfiles):
            if (inompi == True):
                filename = basefilename+".sublevels"+".catalog_groups"
                hdffilename = basefilename+".hdf"+".sublevels"+".catalog_groups"
            else:
                filename = basefilename+".sublevels" + \
                    ".catalog_groups"+"."+str(ifile)
                hdffilename = basefilename+".hdf" + \
                    ".sublevels"+".catalog_groups"+"."+str(ifile)
            if (iverbose):
                print("reading ", filename)
            halofile = open(filename, 'r')
            hdffile = h5py.File(hdffilename, 'w')
            [filenum, numfiles] = halofile.readline().split()
            [numhalos, numtothalos] = halofile.readline().split()
            filenum = int(filenum)
            numfiles = int(numfiles)
            numhalos = np.uint64(numhalos)
            numtothalos = np.uint64(numtothalos)
            # write header info
            hdffile.create_dataset("File_id", data=np.array([filenum]))
            hdffile.create_dataset("Num_of_files", data=np.array([numfiles]))
            hdffile.create_dataset("Num_of_groups", data=np.array([numhalos]))
            hdffile.create_dataset("Total_num_of_groups",
                                   data=np.array([numtothalos]))
            halofile.close()
            if (numhalos > 0):
                cattemp = np.loadtxt(filename, skiprows=2).transpose()
                for ikeys in range(len(fieldnames)):
                    hdffile.create_dataset(fieldnames[ikeys], data=np.array(
                        cattemp[ikeys*numhalos:(ikeys+1)*numhalos], dtype=fieldtype[ikeys]))
            else:
                cattemp = []
                for ikeys in range(len(fieldnames)):
                    hdffile.create_dataset(
                        fieldnames[ikeys], data=np.array([], dtype=fieldtype[ikeys]))
            hdffile.close()


def ConvertASCIICatalogParticleFileToHDF(basefilename, iunbound=0, iseparatesubfiles=0, iverbose=0):
    """
    Reads an ASCII file and converts it to the HDF format for VELOCIraptor files
    """
    inompi = True
    if (iverbose):
        print("reading properties file and converting to hdf",
              basefilename, os.path.isfile(basefilename))
    filename = basefilename+".catalog_particles"
    if (iunbound > 0):
        filename += ".unbound"
    # load header
    if (os.path.isfile(basefilename) == True):
        numfiles = 0
    else:
        filename = basefilename+".catalog_particles"
        if (iunbound > 0):
            filename += ".unbound"
        filename += ".0"
        inompi = False
        if (os.path.isfile(filename) == False):
            print("file not found")
            return []
    byteoffset = 0
    # load ascii file
    halofile = open(filename, 'r')
    # read header information
    [filenum, numfiles] = halofile.readline().split()
    filenum = int(filenum)
    numfiles = int(numfiles)
    [numhalos, numtothalos] = halofile.readline().split()
    numhalos = np.uint64(numhalos)
    numtothalos = np.uint64(numtothalos)
    halofile.close()

    for ifile in range(numfiles):
        if (inompi == True):
            filename = basefilename+".catalog_particles"
            hdffilename = basefilename+".hdf.catalog_particles"
            if (iunbound > 0):
                filename += ".unbound"
                hdffilename += ".unbound"
        else:
            filename = basefilename+".catalog_particles"
            hdffilename = basefilename+".hdf.catalog_particles"
            if (iunbound > 0):
                filename += ".unbound"
                hdffilename += ".unbound"
            filename += "."+str(ifile)
            hdffilename += "."+str(ifile)
        if (iverbose):
            print("reading ", filename)
        halofile = open(filename, 'r')
        hdffile = h5py.File(hdffilename, 'w')
        [filenum, numfiles] = halofile.readline().split()
        [numhalos, numtothalos] = halofile.readline().split()
        filenum = int(filenum)
        numfiles = int(numfiles)
        numhalos = np.uint64(numhalos)
        numtothalos = np.uint64(numtothalos)
        # write header info
        hdffile.create_dataset("File_id", data=np.array([filenum]))
        hdffile.create_dataset("Num_of_files", data=np.array([numfiles]))
        hdffile.create_dataset(
            "Num_of_particles_in_groups", data=np.array([numhalos]))
        hdffile.create_dataset(
            "Total_num_of_particles_in_all_groups", data=np.array([numtothalos]))
        halofile.close()
        if (numhalos > 0):
            cattemp = np.loadtxt(filename, skiprows=2).transpose()
        else:
            cattemp = []
        hdffile.create_dataset(
            "Particle_IDs", data=np.array(cattemp, dtype=np.int64))
        hdffile.close()
    # if subhalos are written in separate files, then read them too
    if (iseparatesubfiles == 1):
        for ifile in range(numfiles):
            if (inompi == True):
                filename = basefilename+".sublevels"+".catalog_particles"
                hdffilename = basefilename+".hdf"+".sublevels"+".catalog_particles"
            else:
                filename = basefilename+".sublevels" + \
                    ".catalog_particles"+"."+str(ifile)
                hdffilename = basefilename+".hdf"+".sublevels" + \
                    ".catalog_particles"+"."+str(ifile)
            if (iverbose):
                print("reading ", filename)
            halofile = open(filename, 'r')
            hdffile = h5py.File(hdffilename, 'w')
            [filenum, numfiles] = halofile.readline().split()
            [numhalos, numtothalos] = halofile.readline().split()
            filenum = int(filenum)
            numfiles = int(numfiles)
            numhalos = np.uint64(numhalos)
            numtothalos = np.uint64(numtothalos)
            # write header info
            hdffile.create_dataset("File_id", data=np.array([filenum]))
            hdffile.create_dataset("Num_of_files", data=np.array([numfiles]))
            hdffile.create_dataset(
                "Num_of_particles_in_groups", data=np.array([numhalos]))
            hdffile.create_dataset(
                "Total_num_of_particles_in_all_groups", data=np.array([numtothalos]))
            halofile.close()
            if (numhalos > 0):
                cattemp = np.loadtxt(filename, skiprows=2).transpose()
            else:
                cattemp = []
            hdffile.create_dataset(
                "Particle_IDs", data=np.array(cattemp, dtype=np.int64))
            hdffile.close()


def ConvertASCIICatalogParticleTypeFileToHDF(basefilename, iunbound=0, iseparatesubfiles=0, iverbose=0):
    """
    Reads an ASCII file and converts it to the HDF format for VELOCIraptor files
    """
    inompi = True
    if (iverbose):
        print("reading properties file and converting to hdf",
              basefilename, os.path.isfile(basefilename))
    filename = basefilename+".catalog_parttypes"
    if (iunbound > 0):
        filename += ".unbound"
    # load header
    if (os.path.isfile(basefilename) == True):
        numfiles = 0
    else:
        filename = basefilename+".catalog_parttypes"
        if (iunbound > 0):
            filename += ".unbound"
        filename += ".0"
        inompi = False
        if (os.path.isfile(filename) == False):
            print("file not found")
            return []
    byteoffset = 0
    # load ascii file
    halofile = open(filename, 'r')
    # read header information
    [filenum, numfiles] = halofile.readline().split()
    filenum = int(filenum)
    numfiles = int(numfiles)
    [numhalos, numtothalos] = halofile.readline().split()
    numhalos = np.uint64(numhalos)
    numtothalos = np.uint64(numtothalos)
    halofile.close()

    for ifile in range(numfiles):
        if (inompi == True):
            filename = basefilename+".catalog_parttypes"
            hdffilename = basefilename+".hdf.catalog_parttypes"
            if (iunbound > 0):
                filename += ".unbound"
                hdffilename += ".unbound"
        else:
            filename = basefilename+".catalog_parttypes"
            hdffilename = basefilename+".hdf.catalog_parttypes"
            if (iunbound > 0):
                filename += ".unbound"
                hdffilename += ".unbound"
            filename += "."+str(ifile)
            hdffilename += "."+str(ifile)
        if (iverbose):
            print("reading ", filename)
        halofile = open(filename, 'r')
        hdffile = h5py.File(hdffilename, 'w')
        [filenum, numfiles] = halofile.readline().split()
        [numhalos, numtothalos] = halofile.readline().split()
        filenum = int(filenum)
        numfiles = int(numfiles)
        numhalos = np.uint64(numhalos)
        numtothalos = np.uint64(numtothalos)
        # write header info
        hdffile.create_dataset("File_id", data=np.array([filenum]))
        hdffile.create_dataset("Num_of_files", data=np.array([numfiles]))
        hdffile.create_dataset(
            "Num_of_particles_in_groups", data=np.array([numhalos]))
        hdffile.create_dataset(
            "Total_num_of_particles_in_all_groups", data=np.array([numtothalos]))
        halofile.close()
        if (numhalos > 0):
            cattemp = np.loadtxt(filename, skiprows=2).transpose()
        else:
            cattemp = []
        hdffile.create_dataset(
            "Particle_types", data=np.array(cattemp, dtype=np.int64))
        hdffile.close()
    # if subhalos are written in separate files, then read them too
    if (iseparatesubfiles == 1):
        for ifile in range(numfiles):
            if (inompi == True):
                filename = basefilename+".sublevels"+".catalog_parttypes"
                hdffilename = basefilename+".hdf"+".sublevels"+".catalog_parttypes"
            else:
                filename = basefilename+".sublevels" + \
                    ".catalog_parttypes"+"."+str(ifile)
                hdffilename = basefilename+".hdf"+".sublevels" + \
                    ".catalog_parttypes"+"."+str(ifile)
            if (iverbose):
                print("reading ", filename)
            halofile = open(filename, 'r')
            hdffile = h5py.File(hdffilename, 'w')
            [filenum, numfiles] = halofile.readline().split()
            [numhalos, numtothalos] = halofile.readline().split()
            filenum = int(filenum)
            numfiles = int(numfiles)
            numhalos = np.uint64(numhalos)
            numtothalos = np.uint64(numtothalos)
            # write header info
            hdffile.create_dataset("File_id", data=np.array([filenum]))
            hdffile.create_dataset("Num_of_files", data=np.array([numfiles]))
            hdffile.create_dataset(
                "Num_of_particles_in_groups", data=np.array([numhalos]))
            hdffile.create_dataset(
                "Total_num_of_particles_in_all_groups", data=np.array([numtothalos]))
            halofile.close()
            if (numhalos > 0):
                cattemp = np.loadtxt(filename, skiprows=2).transpose()
            else:
                cattemp = []
            hdffile.create_dataset(
                "Particle_types", data=np.array(cattemp, dtype=np.int64))
            hdffile.close()


def ConvertASCIIToHDF(basefilename, iseparatesubfiles=0, itype=0, iverbose=0):
    ConvertASCIIPropertyFileToHDF(basefilename, iseparatesubfiles, iverbose)
    ConvertASCIICatalogGroupsFileToHDF(
        basefilename, iseparatesubfiles, iverbose)
    ConvertASCIICatalogParticleFileToHDF(
        basefilename, 0, iseparatesubfiles, iverbose)
    ConvertASCIICatalogParticleFileToHDF(
        basefilename, 1, iseparatesubfiles, iverbose)
    if (itype == 1):
        ConvertASCIICatalogParticleTypeFileToHDF(
            basefilename, 0, iseparatesubfiles, iverbose)
        ConvertASCIICatalogParticleTypeFileToHDF(
            basefilename, 1, iseparatesubfiles, iverbose)
