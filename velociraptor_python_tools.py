# Make backwards compatible with python 2, ignored in python 3
from __future__ import print_function

import sys
import os
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


def ReadPropertyFile(basefilename, ibinary=0, iseparatesubfiles=0, iverbose=0, desiredfields=[], isiminfo=True, iunitinfo=True):
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

    start = time.clock()
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
            print("file not found")
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
    # load associated simulation info, time and units
    if (isiminfo):
        siminfoname = basefilename+".siminfo"
        siminfo = open(siminfoname, 'r')
        catalog['SimulationInfo'] = dict()
        for l in siminfo:
            d = l.strip().split(' : ')
            catalog['SimulationInfo'][d[0]] = float(d[1])
        siminfo.close()
    if (iunitinfo):
        unitinfoname = basefilename+".units"
        unitinfo = open(unitinfoname, 'r')
        catalog['UnitInfo'] = dict()
        for l in unitinfo:
            d = l.strip().split(' : ')
            catalog['UnitInfo'][d[0]] = float(d[1])
        unitinfo.close()

    if (iverbose):
        print("done reading properties file ", time.clock()-start)
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
    start = time.clock()
    tree = []
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
        snaptreelist = open(treefilename, 'r')
        numsnap = sum(1 for line in snaptreelist)
        snaptreelist.close()
    else:
        print("Unknown format, returning null")
        numsnap = 0
        return tree

    tree = [{"haloID": [], "Num_progen": [], "Progen": []}
            for i in range(numsnap)]
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

        snaptreelist = open(treefilename, 'r')
        # read the first file, get number of snaps from hdf file
        snaptreename = snaptreelist.readline().strip()+".tree"
        treedata = h5py.File(snaptreename, "r")
        numsnaps = treedata.attrs['Number_of_snapshots']
        treedata.close()
        snaptreelist.close()

        snaptreelist = open(treefilename, 'r')
        for snap in range(numsnaps):
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
        print("done reading tree file ", time.clock()-start)
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
                                 ireducemem=True):
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
    start = time.clock()
    tree = []
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
    # hdf format, input file is a list of filenames
    elif(ibinary == 2):
        if (iverbose):
            print("Reading HDF5 input")
        snaptreelist = open(treefilename, 'r')
        numsnap = sum(1 for line in snaptreelist)
        snaptreelist.close()
    else:
        print("Unknown format, returning null")
        numsnap = 0
        return tree

    tree = [{"haloID": [], "Num_descen": [], "Descen": [], "Rank": []}
            for i in range(numsnap)]
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
        # read the first file, get number of snaps from hdf file
        snaptreename = snaptreelist.readline().strip()+".tree"
        treedata = h5py.File(snaptreename, "r")
        numsnaps = treedata.attrs['Number_of_snapshots']
        treedata.close()
        snaptreelist.close()
        snaptreelist = open(treefilename, 'r')
        snaptreenames=[[] for snap in range(numsnap)]
        for snap in range(numsnap):
            snaptreenames[snap] = snaptreelist.readline().strip()+".tree"
        snaptreelist.close()

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
                print("Data contains ", numhalos, "halos and will likley minimum ", memsize/1024.**3.0, "GB of memory")
            else:
                print("Contains ", numhalos, "halos and will likley minimum ", memsize/1024.**3.0, "GB of memory")
                print("Plus overhead to store list of arrays, with likely minimum of ",100*numhalos/1024**3.0, "GB of memory ")
            treedata.close()
        for snap in range(numsnap):
            snaptreename = snaptreenames[snap]

            if (iverbose):
                print("Reading", snaptreename)
            treedata = h5py.File(snaptreename, "r")
            tree[snap]["haloID"] = np.array(treedata["ID"])
            tree[snap]["Num_descen"] = np.array(treedata["NumDesc"],np.uint16)
            numhalos=tree[snap]["haloID"].size
            if(inpart):
                tree[snap]["Npart"] = np.asarray(treedata["Npart"],np.int32)

            # See if the dataset exits
            if("DescOffsets" in treedata.keys()):

                # Find the indices to split the array
                if (ireducemem):
                    tree[snap]["_Offsets"] = np.array(treedata["DescOffsets"],dtype=np.uint64)
                else:
                    descenoff=np.array(treedata["DescOffsets"],dtype=np.uint64)
                    split = np.add(descenoff, tree[snap]["Num_descen"], dtype=np.uint64, casting="unsafe")[:-1]
                    descenoff=None

                # Read in the data splitting it up as reading it in
                # if reducing memory then store all the values in the _ keys
                # and generate class that returns the appropriate subchunk as an array when using the [] operaotor
                # otherwise generate lists of arrays
                if (ireducemem):
                    tree[snap]["_Ranks"] = np.array(treedata["Ranks"],dtype=np.int16)
                    tree[snap]["_Descens"] = np.array(treedata["Descendants"],dtype=np.uint64)
                    tree[snap]["Rank"] = MinStorageList(tree[snap]["Num_descen"],tree[snap]["_Offsets"],tree[snap]["_Ranks"])
                    tree[snap]["Descen"] = MinStorageList(tree[snap]["Num_descen"],tree[snap]["_Offsets"],tree[snap]["_Descens"])
                else:
                    tree[snap]["Rank"] = np.split(np.array(treedata["Ranks"],dtype=np.uint16), split)
                    tree[snap]["Descen"] = np.split(np.array(treedata["Descendants"],dtype=np.uint64), split)

                if(inpart):
                    if (ireducemem):
                        tree[snap]["_Npart_descens"] = np.array(treedata["DescenNpart"],np.float32)
                        tree[snap]["Npart_descen"] = MinStorageList(tree[snap]["Num_descen"],tree[snap]["_Offsets"],tree[snap]["_Npart_descens"])
                    else:
                        tree[snap]["Npart_descen"] = np.split(np.array(treedata["DescenNpart"],np.int32), split)
                if(imerit):
                    if (ireducemem):
                        tree[snap]["_Merits"] = np.array(treedata["Merits"],np.float32)
                        tree[snap]["Merit"] = MinStorageList(tree[snap]["Num_descen"],tree[snap]["_Offsets"],tree[snap]["_Merits"])
                    else:
                        tree[snap]["Merit"] = np.split(np.array(treedata["Merits"],np.float32), split)
                #if reducing stuff down to best ranks, then only keep first descendant
                #unless also reading merit and then keep first descendant and all other descendants that are above a merit limit
                if (ireducedtobestranks==True and ireducemem==False):
                    halolist = np.where(tree[snap]["Num_descen"]>1)[0]
                    if (iverbose):
                        print('Reducing memory needed. At snap ', snap, ' with %d total halos and alter %d halos. '% (len(tree[snap]['Num_descen']), len(halolist)))
                        print(np.percentile(tree[snap]['Num_descen'][halolist],[50.0,99.0]))
                    for ihalo in halolist:
                        numdescen = 1
                        if (imerit):
                            numdescen = np.int32(np.max([1,np.argmax(tree[snap]["Merit"][ihalo]<meritlimit)]))
                        tree[snap]["Num_descen"][ihalo] = numdescen
                        tree[snap]["Descen"][ihalo] = np.array([tree[snap]["Descen"][ihalo][:numdescen]])
                        tree[snap]["Rank"][ihalo] = np.array([tree[snap]["Rank"][ihalo][:numdescen]])
                        if (imerit):
                            tree[snap]["Merit"][ihalo] = np.array([tree[snap]["Merit"][ihalo][:numdescen]])
                        if (inpart):
                            tree[snap]["Npart_descen"][ihalo] = np.array([tree[snap]["Npart_descen"][ihalo][:numdescen]])
                split=None
            treedata.close()

    if (iverbose):
        print("done reading tree file ", time.clock()-start)
    return tree


def ReadHaloPropertiesAcrossSnapshots(numsnaps, snaplistfname, inputtype, iseperatefiles, iverbose=0, desiredfields=[]):
    """
    read halo data from snapshots listed in file with snaplistfname file name
    """
    halodata = [dict() for j in range(numsnaps)]
    ngtot = [0 for j in range(numsnaps)]
    atime = [0 for j in range(numsnaps)]
    start = time.clock()
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
    print("data read in ", time.clock()-start)
    return halodata, ngtot, atime


def ReadCrossCatalogList(fname, meritlim=0.1, iverbose=0):
    """
    Reads a cross catalog produced by halomergertree,
    also allows trimming of cross catalog using a higher merit threshold than one used to produce catalog
    """
    return []
    """
    start = time.clock()
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
        print("done reading cross catalog ", time.clock()-start)
    return pdata
    """


def ReadSimInfo(basefilename):
    """
    Reads in the information in .siminfo and returns it as a dictionary
    """

    filename = basefilename + ".siminfo"

    if (os.path.isfile(filename) == False):
        print("file not found")
        return []

    cosmodata = {}
    siminfofile = open(filename, "r")
    line = siminfofile.readline().strip().split(" : ")
    while(line[0] != ""):
        cosmodata[line[0]] = float(line[1])
        line = siminfofile.readline().strip().split(" : ")
    siminfofile.close()
    return cosmodata


def ReadUnitInfo(basefilename):
    """
    Reads in the information in .units and returns it as a dictionary
    """

    filename = basefilename + ".units"

    if (os.path.isfile(filename) == False):
        print("file not found")
        return []

    unitdata = {}
    unitsfile = open(filename, "r")
    line = unitsfile.readline().strip().split(" : ")
    while(line[0] != ""):
        unitdata[line[0]] = float(line[1])
        line = unitsfile.readline().strip().split(" : ")
    unitsfile.close()
    return unitdata


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
                    tdata = np.uint16(pfile["Particle_Types"])
                    utdata = np.uint16(upfile["Particle_Types"])
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


def ReadSOParticleDataFile(basefilename, ibinary=0, iverbose=0, binarydtype=np.int64):
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
    if (iverbose):
        print("SO lists contains ", numtotSO, " regions containing total of ",
              numtotparts, " in ", numfiles, " files")
    if (numtotSO == 0):
        return particledata
    particledata['Npart'] = np.zeros(numtotSO, dtype=np.uint64)
    particledata['Particle_IDs'] = [[] for i in range(numtotSO)]

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
        # binary
        elif (ibinary == 1):
            gfile = open(filename, 'rb')
            np.fromfile(gfile, dtype=np.int32, count=2)
            [numSO, foo] = np.fromfile(gfile, dtype=np.uint64, count=2)
            [numparts, foo] = np.fromfile(gfile, dtype=np.uint64, count=2)
            numingroup = np.fromfile(gfile, dtype=binarydtype, count=numSO)
            offset = np.fromfile(gfile, dtype=binarydtype, count=numSO)
            piddata = np.fromfile(gfile, dtype=binarydtype, count=numparts)
            gfile.close()
        # hdf
        elif (ibinary == 2):
            gfile = h5py.File(filename, 'r')
            numSO = np.uint64(gfile["Num_of_SO_regions"][0])
            numingroup = np.uint64(gfile["SO_size"])
            offset = np.uint64(gfile["Offset"])
            piddata = np.int64(gfile["Particle_IDs"])
            gfile.close()

        # now with data loaded, process it to produce data structure
        particledata['Npart'][counter:counter+numSO] = numingroup
        for i in range(numSO):
            particledata['Particle_IDs'][int(
                i+counter)] = np.array(piddata[offset[i]:offset[i]+numingroup[i]])
        counter += numSO

    return particledata


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
    start = time.clock()
    if (iverbose):
        print("setting hierarchy")
    numhalos = len(halodata["npart"])
    subhaloindex = np.where(halodata["hostHaloID"] != -1)
    lensub = len(subhaloindex[0])
    haloindex = np.where(halodata["hostHaloID"] == -1)
    lenhal = len(haloindex[0])
    halohierarchy = [[] for k in range(numhalos)]
    if (iverbose):
        print("prelims done ", time.clock()-start)
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
        print("hierarchy set in read in ", time.clock()-start)
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
    for k in range(numsnaps):
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

    totstart = time.clock()

    if (iparallel == 1):
        # need to copy halodata as this will be altered
        if (iverbose > 0):
            print("copying halo")
        start = time.clock()
        mphalodata = manager.list([manager.dict(halodata[k])
                                   for k in range(numsnaps)])
        if (iverbose > 0):
            print("done", time.clock()-start)

    for istart in range(numsnaps):
        if (iverbose > 0):
            print("Starting from halos at ", istart, "with", numhalos[istart])
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
            # now for each chunk run a set of proceses
            for j in range(nchunks):
                start = time.clock()
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
                    p.start()
                    count += 1
                for p in processes:
                    # join thread and see if still active
                    p.join()
                if (iverbose > 1):
                    print((offset+j*nthreads*chunksize) /
                          float(numhalos[istart]), " done in", time.clock()-start)
        # otherwise just single
        else:
            # if first time entering non parallel section copy data back from parallel manager based structure to original data structure
            # as parallel structures have been updated
            if (iparallel == 1):
                #tree = [dict(mptree[k]) for k in range(numsnaps)]
                halodata = [dict(mphalodata[k]) for k in range(numsnaps)]
                # set the iparallel flag to 0 so that all subsequent snapshots (which should have fewer objects) not run in parallel
                # this is principly to minimize the amount of copying between manager based parallel structures and the halo/tree catalogs
                iparallel = 0
            start = time.clock()
            chunksize = max(int(0.10*numhalos[istart]), 10)
            for j in range(numhalos[istart]):
                # start at this snapshot
                #start = time.clock()
                TraceMainProgen(istart, j, numsnaps, numhalos,
                                halodata, tree, TEMPORALHALOIDVAL)
                if (j % chunksize == 0 and j > 0):
                    if (iverbose > 1):
                        print(
                            "done", j/float(numhalos[istart]), "in", time.clock()-start)
                    start = time.clock()
    if (iverbose > 0):
        print("done with first bit")
    # now have walked all the main branches and set the root head, head and tail values
    # and can set the root tail of all halos. Start at end of the tree and move in reverse setting the root tail
    # of a halo's head so long as that halo's tail is the current halo (main branch)
    for istart in range(numsnaps-1, -1, -1):
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
    print("Done building", time.clock()-totstart)


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
    for k in range(numsnaps):
        halodata[k]['Head'] = np.zeros(numhalos[k], dtype=np.int64)
        halodata[k]['Tail'] = np.zeros(numhalos[k], dtype=np.int64)
        halodata[k]['HeadSnap'] = np.zeros(numhalos[k], dtype=np.int32)
        halodata[k]['TailSnap'] = np.zeros(numhalos[k], dtype=np.int32)
        halodata[k]['RootHead'] = np.zeros(numhalos[k], dtype=np.int64)
        halodata[k]['RootTail'] = np.zeros(numhalos[k], dtype=np.int64)
        halodata[k]['RootHeadSnap'] = np.zeros(numhalos[k], dtype=np.int32)
        halodata[k]['RootTailSnap'] = np.zeros(numhalos[k], dtype=np.int32)
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

    totstart = time.clock()
    start0=time.clock()

    if (ireverseorder):
        snaplist = range(numsnaps-1, -1, -1)
    else:
        snaplist = range(numsnaps)
    for istart in snaplist:
        start2=time.clock()
        if (iverbose > 0):
            print('starting head/tail at snapshot ', istart, ' containing ', numhalos[istart], 'halos')
        if (numhalos[istart] == 0): continue
        #set tails and root tails if necessary
        wdata = np.where(halodata[istart]['Tail'] == 0)[0]
        numareroottails = wdata.size
        if (iverbose > 0):
            print(numareroottails,' halos are root tails ')
        if (numareroottails > 0):
            halodata[istart]['Tail'][wdata] = np.array(halodata[istart]['ID'][wdata],copy=True)
            halodata[istart]['RootTail'][wdata] = np.array(halodata[istart]['ID'][wdata],copy=True)
            halodata[istart]['TailSnap'][wdata] = istart*np.ones(wdata.size, dtype=np.int32)
            halodata[istart]['RootTailSnap'][wdata] = istart*np.ones(wdata.size, dtype=np.int32)
        #init heads to ids
        halodata[istart]['Head'] = np.array(halodata[istart]['ID'],copy=True)
        halodata[istart]['HeadSnap'] = istart*np.ones(numhalos[istart])
        #find all halos that have descendants and set there heads
        if (istart == numsnaps-1):
            halodata[istart]['RootHead'] = np.array(halodata[istart]['ID'],copy=True)
            halodata[istart]['RootHeadSnap'] = istart*np.ones(numhalos[istart], dtype=np.int32)
            continue
        wdata = None
        descencheck=(tree[istart]['Num_descen']>0)
        wdata=np.where(descencheck)[0]
        numwithdescen = wdata.size
        if (iverbose > 0):
            print(numwithdescen, 'have descendants')
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
                    #if tail was assigned then need to compare merits and designed which one to use
                    else:
                        #if can compare merits
                        if (imerit):
                            curMerit = activemerits[i]
                            prevTailIndex = np.int64(halodata[isnap]['Tail'][idescenindex] % TEMPORALHALOIDVAL - 1)
                            prevTailSnap = halodata[isnap]['TailSnap'][idescenindex]
                            compMerit = tree[prevTailSnap]['Merit'][prevTailIndex][0]
                            if (curMerit > compMerit):
                                halodata[prevTailSnap]['HeadRank'][prevTailIndex]+=1
                                halodata[isnap]['Tail'][idescenindex] = halodata[istart]['ID'][index]
                                halodata[isnap]['RootTail'][idescenindex] = halodata[istart]['RootTail'][index]
                                halodata[isnap]['TailSnap'][idescenindex] = istart
                                halodata[isnap]['RootTailSnap'][idescenindex] = halodata[istart]['RootTailSnap'][index]
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
        wdata = None
        descencheck = None
        if (iverbose > 0):
            print('finished in', time.clock()-start2)
    if (iverbose > 0):
        print("done with first bit, setting the main branches walking backward",time.clock()-start0)
    # now have walked all the main branches and set the root tail, head and tail values
    # in case halo data is with late times at beginning need to process items in reverse
    if (ireverseorder):
        snaplist = range(numsnaps)
    else:
        snaplist = range(numsnaps-1, -1, -1)
    # first set root heads of main branches
    for istart in snaplist:
        if (numhalos[istart] == 0):
            continue
        wdata = np.where((halodata[istart]['RootHead'] != 0))[0]
        numactive=wdata.size
        if (iverbose > 0):
            print('Setting root heads at ', istart, 'halos', numhalos[istart], 'active', numactive)
        if (numactive == 0):
            continue

        haloidarray = halodata[istart]['Tail'][wdata]
        haloindexarray = np.array(haloidarray % TEMPORALHALOIDVAL -1, dtype=np.int64)
        halosnaparray = np.array((haloidarray - haloindexarray - np.int64(1)) / TEMPORALHALOIDVAL, dtype=np.int32)

        if (ireverseorder):
            halosnaparray = numsnaps - 1 - halosnaparray
        # go to root tails and walk the main branch
        for i in np.arange(numactive,dtype=np.int64):
            halodata[halosnaparray[i]]['RootHead'][haloindexarray[i]]=halodata[istart]['RootHead'][wdata[i]]
            halodata[halosnaparray[i]]['RootHeadSnap'][haloindexarray[i]]=halodata[istart]['RootHeadSnap'][wdata[i]]
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
            # now set the root head for all the progenitors of this object
            while (True):
                halodata[halosnap]['RootHead'][haloindex] = roothead
                halodata[halosnap]['RootHeadSnap'][haloindex] = rootsnap
                if (haloid == halodata[halosnap]['Tail'][haloindex]):
                    break
                haloid = halodata[halosnap]['Tail'][haloindex]
                halosnap = halodata[halosnap]['TailSnap'][haloindex]
                haloindex = np.int64(haloid % TEMPORALHALOIDVAL - 1)
        rankedhalos = None
        rankedhaloindex = None
        maindescen = None
        maindescenindex = None
        maindescensnaporder = None

    print("Done building", time.clock()-totstart)


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
    for j in range(numsnaps):
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
        pos = [[]for j in range(numsnaps)]
        pos_tree = [[]for j in range(numsnaps)]
        start = time.clock()
        if (iverbose):
            print("tree build")
        for j in range(numsnaps):
            if (numhalos[j] > 0):
                boxval = boxsize*atime[j]/hval
                pos[j] = np.transpose(np.asarray(
                    [halodata[j]["Xc"], halodata[j]["Yc"], halodata[j]["Zc"]]))
                pos_tree[j] = spatial.cKDTree(pos[j], boxsize=boxval)
        if (iverbose):
            print("done ", time.clock()-start)
    # else assume tree has been passed
    for j in range(numsnaps):
        if (numhalos[j] == 0):
            continue
        # at snapshot look at all haloes that have not had a major merger set
        # note that only care about objects with certain number of particles
        partcutwdata = np.where(halodata[j]["npart"] >= NPARTCUT)
        mergercut = np.where(halodata[j]["LastMergerRatio"][partcutwdata] < 0)
        hids = np.asarray(halodata[j]["ID"][partcutwdata]
                          [mergercut], dtype=np.uint64)
        start = time.clock()
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
            print("Done snap", j, time.clock()-start)


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
    for j in range(numsnaps):
        # store id and snap and mass of last major merger and while we're at it, store number of major mergers
        halodata[j]["NextSubhalo"] = copy.deepcopy(halodata[j]["ID"])
        halodata[j]["PreviousSubhalo"] = copy.deepcopy(halodata[j]["ID"])
    # iterate over all host halos and set their subhalo links
    start = time.clock()
    nthreads = 1
    if (iparallel):
        manager = mp.Manager()
        nthreads = int(min(mp.cpu_count(), numsnaps))
        print("Number of threads is ", nthreads)
    for j in range(0, numsnaps, nthreads):
        start2 = time.clock()
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
                print("Done snaps", j, "to", j+nthreads, time.clock()-start2)
                sys.stdout.flush()

        else:
            generate_sublinks(numhalos[j], halodata[j], iverbose)
            if (iverbose):
                print("Done snap", j, time.clock()-start2)
                sys.stdout.flush()
    print("Done subhalolinks ", time.clock()-start)
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

    for j in range(numsnaps):
        # store id and snap and mass of last major merger and while we're at it, store number of major mergers
        halodata[j]["LeftTail"] = copy.deepcopy(halodata[j]["ID"])
        halodata[j]["RightTail"] = copy.deepcopy(halodata[j]["ID"])
        # alias the data
        halodata[j]["PreviousProgenitor"] = halodata[j]["LeftTail"]
        halodata[j]["NextProgenitor"] = halodata[j]["RightTail"]
    # move backward in time and identify all unique heads
    start = time.clock()
    if (ireversesnaporder):
        snaplist = range(1, numsnaps)
    else:
        snaplist = range(numsnaps-2, -1, -1)
    for j in snaplist:
        start2 = time.clock()
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
            print("Done snap", j, time.clock()-start2)
            sys.stdout.flush()
    print("Done progenitor links ", time.clock()-start)
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

    # initialize the dictionaries
    for j in range(numsnaps):
        # store id and snap and mass of last major merger and while we're at it, store number of major mergers
        halodata[j]["ForestID"] = np.ones(numhalos[j], dtype=np.int64)*-1
        halodata[j]["ForestLevel"] = np.ones(numhalos[j], dtype=np.int32)*-1
    # built KD tree to quickly search for near neighbours. only build if not passed.
    if (ispatialintflag):
        start = time.clock()
        boxsize = cosmo['BoxSize']
        hval = cosmo['Hubble_param']
        if (len(pos_tree) == 0):
            pos = [[]for j in range(numsnaps)]
            pos_tree = [[]for j in range(numsnaps)]
            start = time.clock()
            if (iverbose):
                print("KD tree build")
            for j in range(numsnaps):
                if (numhalos[j] > 0):
                    boxval = boxsize*atime[j]/hval
                    pos[j] = np.transpose(np.asarray(
                        [halodata[j]["Xc"], halodata[j]["Yc"], halodata[j]["Zc"]]))
                    pos_tree[j] = spatial.cKDTree(pos[j], boxsize=boxval)
            if (iverbose):
                print("done ", time.clock()-start)

    # now start marching backwards in time from root heads
    # identifying all subhaloes that have every been subhaloes for long enough
    # and all progenitors and group them together into the same forest id
    forestidval = 1
    start = time.clock()
    # for j in range(numsnaps):
    # set the direction of how the data will be processed
    if (ireversesnaporder):
        snaplist = np.arange(0, numsnaps, dtype=np.int32)
    else:
        snaplist = np.arange(numsnaps-1, -1, -1)
    # first pass assigning forests based on FOF and subs
    offset = 0
    start2 = time.clock()
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
        [halodata[i]['ForestID'] for i in range(numsnaps)]), return_counts=True)
    numforests = len(ForestIDs)
    maxforest = np.max(ForestSize)
    print('finished first pass', time.clock()-start2,
          'have ', numforests, 'initial forests',
          'with largest forest containing ', maxforest, '(sub)halos')
    sys.stdout.flush()

    #ForestSizeStats = dict(zip(ForestIDs, ForestSize))
    #store the a map of forest ids that will updated
    ForestMap = dict(zip(ForestIDs, ForestIDs))
    # free memory
    ForestIDs = ForestSize = None

    # now proceed to find new mappings
    start1 = time.clock()
    numloops = 0
    while (True):
        newforests = 0
        start2 = time.clock()
        if (iverbose):
            print('walking forward in time looking at descendants')
            sys.stdout.flush()
        if (ireversesnaporder):
            snaplist = np.arange(0, numsnaps-1, dtype=np.int32)[::-1]
        else:
            snaplist = np.arange(numsnaps-1, 0, -1)[::-1]
        for j in snaplist:
            if (numhalos[j] == 0):
                continue
            start3 = time.clock()
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
                # find all descendants of currently active halos by finding those whose tails point to snapshot of the descendant list
                descens = np.where(
                    np.int32(halodata[k]['Tail']/TEMPORALHALOIDVAL) == j)[0]
                if (len(descens) == 0):
                    continue
                # if first loop passed, then some amount of forest ids have been updated and can limit search to those that don't match
                if (numloops >= 1):
                    wdata = np.where(halodata[k]['ForestID'][descens] != halodata[j]['ForestID'][np.int64(
                        halodata[k]['Tail'][descens] % TEMPORALHALOIDVAL-1)])
                    if (len(wdata[0]) == 0):
                        continue
                    descens = descens[wdata]
                    wdata = None

                # process snap to update forest id map
                tailindexarray = np.array((halodata[k]['Tail'][descens] % TEMPORALHALOIDVAL-1), dtype=np.int64,copy=True)
                for icount in range(descens.size):
                    idescen = descens[icount]
                    itail = tailindexarray[icount]
                    curforest = halodata[k]['ForestID'][idescen]
                    refforest = halodata[j]['ForestID'][itail]
                    # it is possible that after updating can have the descedants forest id match its progenitor forest id so do nothing if this is the case
                    if (ForestMap[curforest] == ForestMap[refforest]):
                        continue
                    # if ref forest is smaller update the mapping
                    if (ForestMap[curforest] > ForestMap[refforest]):
                        ForestMap[curforest]=ForestMap[refforest]
                        newforests += 1
                        incforests += 1
                    else :
                        ForestMap[refforest]=ForestMap[curforest]
                        newforests += 1
                        incforests += 1
        if (iverbose):
            print('done walking forward, found  ', newforests, ' new forest links at ',
                  numloops, ' loop in a time of ', time.clock()-start2)
            sys.stdout.flush()
        if (newforests == 0):
            break
        # update forest ids using map
        start2 = time.clock()
        for j in snaplist:
            if (numhalos[j] == 0):
                continue
            for ihalo in range(numhalos[j]):
                halodata[j]['ForestID'][ihalo]=ForestMap[halodata[j]['ForestID'][ihalo]]
        if (iverbose):
            print('Finished remapping in', time.clock()-start2)
            sys.stdout.flush()
        numloops += 1

    print('Done linking between forests in %d in a time of %f' %
          (numloops, time.clock()-start1))
    sys.stdout.flush()

    # get the size of each forest
    ForestIDs, ForestSize = np.unique(np.concatenate(
        [halodata[i]['ForestID'] for i in range(numsnaps)]), return_counts=True)
    numforests = ForestIDs.size
    maxforest = np.max(ForestSize)
    print('Forest consists of ', numforests, 'with largest', maxforest)

    ForestSizeStats = dict()
    ForestSizeStats['AllSnaps'] = dict(zip(ForestIDs, ForestSize))
    ForestSizeStats['Number_of_forests'] = numforests
    ForestSizeStats['Largest_forest_size'] = maxforest
    ForestSizeStats['Snapshots'] = dict()
    for i in range(numsnaps):
        ForestSizeStats['Snapshots']['Snap_%03d' %
                                     i] = np.zeros(numforests, dtype=np.int64)
        if (numhalos[i] == 0):
            continue
        activeforest, counts = np.unique(
            halodata[i]['ForestID'], return_counts=True)
        ForestSizeStats['Snapshots']['Snap_%03d' %
                                     i][np.where(np.in1d(ForestIDs, activeforest))] = counts

    start2 = time.clock()
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
            for j in range(numsnaps):
                halodata[j]["ForestID"] = np.ones(numhalos[j], dtype=np.int64)*-1
                halodata[j]["ForestLevel"] = np.ones(numhalos[j], dtype=np.int32)*-1
            return []

    # then return this
    print("Done generating forest", time.clock()-start)
    sys.stdout.flush()
    return ForestSizeStats


"""
Adjust halo catalog for period, comoving coords, etc
"""


def AdjustforPeriod(numsnaps, numhalos, boxsize, hval, atime, halodata, icomove=0):
    """
    Map halo positions from 0 to box size
    """
    for i in range(numsnaps):
        if (icomove):
            boxval = boxsize/hval
        else:
            boxval = boxsize*atime[i]/hval
        wdata = np.where(halodata[i]["Xc"] < 0)
        halodata[i]["Xc"][wdata] += boxval
        wdata = np.where(halodata[i]["Yc"] < 0)
        halodata[i]["Yc"][wdata] += boxval
        wdata = np.where(halodata[i]["Zc"] < 0)
        halodata[i]["Zc"][wdata] += boxval

        wdata = np.where(halodata[i]["Xc"] > boxval)
        halodata[i]["Xc"][wdata] -= boxval
        wdata = np.where(halodata[i]["Yc"] > boxval)
        halodata[i]["Yc"][wdata] -= boxval
        wdata = np.where(halodata[i]["Zc"] > boxval)
        halodata[i]["Zc"][wdata] -= boxval


def AdjustComove(itocomovefromphysnumsnaps, numsnaps, numhalos, atime, halodata, igas=0, istar=0):
    """
    Convert distances to/from physical from/to comoving
    """
    for i in range(numsnaps):
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
    treebuildergrp.attrs["Version"] = descripdata["TreeBuilder_version"]
    treebuildergrp.attrs["Temporal_linking_length"] = descripdata["Temporal_linking_length"]
    treebuildergrp.attrs["Temporal_halo_id_value"] = descripdata["Temporal_halo_id_value"]


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

    for i in range(numsnaps):

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
    treebuildergrp.attrs["Version"] = descripdata["Tree_version"]
    treebuildergrp.attrs["Temporal_linking_length"] = descripdata["Temporal_linking_length"]

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

    for i in range(numsnaps):
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
    for i in range(numsnaps):
        tdata[count:int(numhalos[i])+count] = halodata[i]["ID"]
        count += int(numhalos[i])
    treegrp.create_dataset("HaloSnapID", data=tdata)
    tdata = np.zeros(tothalos, dtype=np.uint32)
    count = 0
    for i in range(numsnaps):
        tdata[count:int(numhalos[i])+count] = i
        count += int(numhalos[i])
    treegrp.create_dataset("HaloSnapNum", data=tdata)
    tdata = np.zeros(tothalos, dtype=np.uint64)
    count = 0
    for i in range(numsnaps):
        tdata[count:int(numhalos[i])+count] = range(int(numhalos[i]))
        count += int(numhalos[i])
    treegrp.create_dataset("HaloSnapIndex", data=tdata)
    # store progenitors
    tdata = np.zeros(tothalos, dtype=halodata[0]["Tail"].dtype)
    count = 0
    for i in range(numsnaps):
        tdata[count:int(numhalos[i])+count] = halodata[i]["Tail"]
        count += int(numhalos[i])
    treegrp.create_dataset("ProgenitorID", data=tdata)
    tdata = np.zeros(tothalos, dtype=halodata[0]["TailSnap"].dtype)
    count = 0
    for i in range(numsnaps):
        tdata[count:int(numhalos[i])+count] = halodata[i]["TailSnap"]
        count += int(numhalos[i])
    treegrp.create_dataset("ProgenitorSnapnum", data=tdata)
    tdata = np.zeros(tothalos, dtype=np.uint64)
    count = 0
    for i in range(numsnaps):
        tdata[count:int(numhalos[i])+count] = (halodata[i]
                                               ["Tail"] % TEMPORALHALOIDVAL-1)
        count += int(numhalos[i])
    treegrp.create_dataset("ProgenitorIndex", data=tdata)
    # store descendants
    tdata = np.zeros(tothalos, dtype=halodata[0]["Head"].dtype)
    count = 0
    for i in range(numsnaps):
        tdata[count:int(numhalos[i])+count] = halodata[i]["Head"]
        count += int(numhalos[i])
    treegrp.create_dataset("DescendantID", data=tdata)
    tdata = np.zeros(tothalos, dtype=halodata[0]["HeadSnap"].dtype)
    count = 0
    for i in range(numsnaps):
        tdata[count:int(numhalos[i])+count] = halodata[i]["HeadSnap"]
        count += int(numhalos[i])
    treegrp.create_dataset("DescendantSnapnum", data=tdata)
    tdata = np.zeros(tothalos, dtype=np.uint64)
    count = 0
    for i in range(numsnaps):
        tdata[count:int(numhalos[i])+count] = (halodata[i]
                                               ["Head"] % TEMPORALHALOIDVAL-1)
        count += int(numhalos[i])
    treegrp.create_dataset("DescendantIndex", data=tdata)
    # store progenitors
    tdata = np.zeros(tothalos, dtype=halodata[0]["RootTail"].dtype)
    count = 0
    for i in range(numsnaps):
        tdata[count:int(numhalos[i])+count] = halodata[i]["RootTail"]
        count += int(numhalos[i])
    treegrp.create_dataset("RootProgenitorID", data=tdata)
    tdata = np.zeros(tothalos, dtype=halodata[0]["RootTailSnap"].dtype)
    count = 0
    for i in range(numsnaps):
        tdata[count:int(numhalos[i])+count] = halodata[i]["RootTailSnap"]
        count += int(numhalos[i])
    treegrp.create_dataset("RootProgenitorSnapnum", data=tdata)
    tdata = np.zeros(tothalos, dtype=np.uint64)
    count = 0
    for i in range(numsnaps):
        tdata[count:int(numhalos[i])+count] = (halodata[i]
                                               ["RootTail"] % TEMPORALHALOIDVAL-1)
        count += int(numhalos[i])
    treegrp.create_dataset("RootProgenitorIndex", data=tdata)
    # store descendants
    tdata = np.zeros(tothalos, dtype=halodata[0]["RootHead"].dtype)
    count = 0
    for i in range(numsnaps):
        tdata[count:int(numhalos[i])+count] = halodata[i]["RootHead"]
        count += int(numhalos[i])
    treegrp.create_dataset("RootDescendantID", data=tdata)
    tdata = np.zeros(tothalos, dtype=halodata[0]["RootHeadSnap"].dtype)
    count = 0
    for i in range(numsnaps):
        tdata[count:int(numhalos[i])+count] = halodata[i]["RootHeadSnap"]
        count += int(numhalos[i])
    treegrp.create_dataset("RootDescendantSnapnum", data=tdata)
    tdata = np.zeros(tothalos, dtype=np.uint64)
    count = 0
    for i in range(numsnaps):
        tdata[count:int(numhalos[i])+count] = (halodata[i]
                                               ["RootHead"] % TEMPORALHALOIDVAL-1)
        count += int(numhalos[i])
    treegrp.create_dataset("RootDescendantIndex", data=tdata)
    # store number of progenitors
    tdata = np.zeros(tothalos, dtype=np.uint32)
    count = 0
    for i in range(numsnaps):
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
    start = time.clock()
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
    print("read halo data ", time.clock()-start)
    return halodata, numhalos, atime, simdata, unitdata


def WriteWalkableHDFTree(fname, numsnaps, tree, numhalos, halodata, atime,
                         descripdata={'Title': 'Tree catalogue', 'TreeBuilder': 'TreeFrog', 'TreeBuilder_version': 1.2,
                                      'Particle_num_threshold': 20, 'Temporal_linking_length': 1, 'Temporal_halo_id_value':1000000000000}
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
    treebuildergrp = headergrp.create_group("TreeBuilder")
    treebuildergrp.attrs["Name"] = descripdata["TreeBuilder"]
    treebuildergrp.attrs["Version"] = descripdata["TreeBuilder_version"]
    treebuildergrp.attrs["Temporal_linking_length"] = descripdata["Temporal_linking_length"]
    treebuildergrp.attrs["Temporal_halo_id_value"] = descripdata["Temporal_halo_id_value"]

    # now need to create groups for halos and then a group containing tree information
    snapsgrp = hdffile.create_group("Snapshots")
    # tree keys of interest
    halokeys = ["RootHead", "RootHeadSnap", "Head", "HeadSnap", "Tail",
                "TailSnap", "RootTail", "RootTailSnap", "ID", "Num_progen"]

    for i in range(numsnaps):
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
    numsnaps = hdffile['Header'].attrs["NSnaps"]
    #nsnapsearch = ["Header/TreeBuilder"].attrs["Temporal_linking_length"]
    if (iverbose):
        print("number of snaps", numsnaps)
    halodata = [dict() for i in range(numsnaps)]
    for i in range(numsnaps):
        # note that I normally have information in reverse order so that might be something in the units
        if (iverbose):
            print("snap ", i)
        for key in hdffile['Snapshots']['Snap_%03d' % i].keys():
            halodata[i][key] = np.array(
                hdffile['Snapshots']['Snap_%03d' % i][key])
    hdffile.close()
    # , nsnapsearch
    return halodata, numsnaps


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
                            secondaryProgenList
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

    count = 0
    if (halodata[curSnap]['Num_progen'][curIndex] > 1):
        haloIDList.append(curHalo)
    """
    curHost = halodata[curSnap]['hostHaloID'][curIndex]
    if (curHost != -1):
        curHostSnap = np.uint64(curHost / TEMPORALHALOIDVAL)
        curHostIndex = np.uint64(curHost % TEMPORALHALOIDVAL-1)
        if (halodata[curSnap]['Num_progen'][curHostIndex] > 1 and
            halodata[curSnap]['RootHead'][curHostIndex] == curRootHead):
            haloIDList.append(curHost)
    """

    # search backwards in time for any object that might have merged with either object or objects host
    while(curSnap >= searchrange and halodata[curSnap]['Tail'][curIndex] != curHalo):
        curHalo = halodata[curSnap]['Tail'][curIndex]
        curSnap = np.uint64(curHalo/TEMPORALHALOIDVAL)
        curIndex = np.uint64(curHalo % TEMPORALHALOIDVAL-1)
        if (halodata[curSnap]['Num_progen'][curIndex] > 1):
            haloIDList.append(curHalo)
        """
        curHost = halodata[curSnap]['hostHaloID'][curIndex]
        if (curHost != -1):
            curHostSnap = np.uint64(curHost/TEMPORALHALOIDVAL)
            curHostIndex = np.uint64(curHost % TEMPORALHALOIDVAL-1)
            if (halodata[curSnap]['Num_progen'][curHostIndex] > 1 and
                halodata[curSnap]['RootHead'][curHostIndex] == curRootHead):
                haloIDList.append(curHost)
        """

    # with this list of objects, search if any of these objects have
    # secondary progenitors with high Merit_type
    haloIDList = np.unique(np.array(haloIDList, dtype=np.int64))
    #identify possible secondary progenitors of halos of interest
    mergeCheck = (np.in1d(secondaryProgenList['Descen'], haloIDList) *
            (secondaryProgenList['Merit'] >= meritlim)
            )
    mergeCandidateList = np.where(mergeCheck)[0]

    if (iverbose > 1):
        print('halo in phase check general ', haloID, 'with number of candidates', mergeCandidateList.size)

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

def FixBranchPhaseComplexBranchSwapAdjustTree(numsnaps, treedata, halodata, numhalos,
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
    """

    if (iverbose > 1):
        print('Adjusting branch swap merge/fragmentation ',haloID, postmergeHalo, branchfixHalo)

    # store the branch fix point
    branchfixSnap = np.uint64(branchfixHalo/TEMPORALHALOIDVAL)
    branchfixIndex = np.uint64(branchfixHalo % TEMPORALHALOIDVAL-1)
    branchfixHead = halodata[branchfixSnap]['Head'][branchfixIndex]
    branchfixHeadSnap = np.uint64(branchfixHead/TEMPORALHALOIDVAL)
    branchfixHeadIndex = np.uint64(branchfixHead % TEMPORALHALOIDVAL-1)
    # now adjust these points, mergeHalo must have its Head changed,
    # haloID must have tail and RootTail and all its descedants that share the same root tail updated
    # branchfixHalo must have its head changed and all its descedants that share the same root tail updated
    # postmergeHalo must now point to branchfixHalo and update head/tail and also now will point to Head of branchfixHalo
    newroottail = halodata[mergeSnap]['RootTail'][mergeIndex]
    newroottailbranchfix = halodata[branchfixSnap]['RootTail'][branchfixIndex]
    newroottailSnap = halodata[mergeSnap]['RootTailSnap'][mergeIndex]
    newroottailbranchfixSnap = halodata[branchfixSnap]['RootTailSnap'][branchfixIndex]
    oldroottail = halodata[postmergeSnap]['RootTail'][postmergeIndex]
    if (iverbose > 1):
        print('new tails will be ', newroottail, newroottailbranchfix)

    # adjust head tails of object with no progenitor
    if (iverbose > 1):
        print('before fix merge', mergeHalo, halodata[mergeSnap]['Head']
              [mergeIndex], 'no prog', haloID, halodata[haloSnap]['Tail'][haloIndex])
    halodata[mergeSnap]['Head'][mergeIndex] = haloID
    halodata[mergeSnap]['HeadSnap'][mergeIndex] = haloSnap
    halodata[haloSnap]['Tail'][haloIndex] = mergeHalo
    halodata[haloSnap]['TailSnap'][haloIndex] = mergeSnap

    # adjust head tails of branch swap line
    if (iverbose > 1):
        print('before fix branch fix', branchfixHalo, halodata[branchfixSnap]['Head'][branchfixIndex],
              ' post merge', postmergeHalo, halodata[postmergeSnap]['Tail'][postmergeIndex])
    halodata[branchfixSnap]['Head'][branchfixIndex] = postmergeHalo
    halodata[branchfixSnap]['HeadSnap'][branchfixIndex] = postmergeSnap
    halodata[postmergeSnap]['Tail'][postmergeIndex] = branchfixHalo
    halodata[postmergeSnap]['TailSnap'][postmergeIndex] = branchfixSnap
    halodata[postmergeSnap]['Head'][postmergeIndex] = branchfixHead
    halodata[postmergeSnap]['HeadSnap'][postmergeIndex] = branchfixHeadSnap
    halodata[branchfixHeadSnap]['Tail'][branchfixHeadIndex] = postmergeHalo
    halodata[branchfixHeadSnap]['TailSnap'][branchfixHeadIndex] = postmergeSnap

    # update the root tails
    curHalo = haloID
    curSnap = np.uint64(curHalo / TEMPORALHALOIDVAL)
    curIndex = np.uint64(curHalo % TEMPORALHALOIDVAL - 1)
    # while (halodata[curSnap]['RootTail'][curIndex] ==  haloID):
    while (True):
        if (iverbose > 2):
            print('moving up branch to adjust the root tails', curHalo,
                  curSnap, halodata[curSnap]['RootTail'][curIndex], newroottail)
        halodata[curSnap]['RootTail'][curIndex] = newroottail
        halodata[curSnap]['RootTailSnap'][curIndex] = newroottailSnap
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
    curSnap = np.uint64(curHalo/TEMPORALHALOIDVAL)
    curIndex = np.uint64(curHalo % TEMPORALHALOIDVAL - 1)
    # and halodata[curSnap]['Head'][curIndex]!= curHalo
    # while (halodata[curSnap]['RootTail'][curIndex] ==  oldroottail):
    while (True):
        if (iverbose > 2):
            print('moving up fix branch to adjust the root tails', curHalo, curSnap,
                  halodata[curSnap]['RootTail'][curIndex], newroottailbranchfix)
        halodata[curSnap]['RootTail'][curIndex] = newroottailbranchfix
        halodata[curSnap]['RootTailSnap'][curIndex] = newroottailbranchfixSnap
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
                            npartlim, meritlim,
                            TEMPORALHALOIDVAL, iverbose,
                            haloID, haloSnap, haloIndex, haloRootHeadID,
                            mergeHalo
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
    branchfixSwapBranchTail = -1
    branchfixMerit = -1

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
                    treedata[subTailSnap]['Merit'][subTailIndex][wdata] >= meritlim and
                    treedata[subTailSnap]['Merit'][subTailIndex][wdata] > branchfixMerit):
                    branchfixSwapBranch = halodata[haloSnap]['ID'][isub]
                    branchfixSwapBranchTail = subTail
                    branchfixMerit = treedata[subTailSnap]['Merit'][subTailIndex][wdata]
                    branchfixSwapBranchMeritIndex = wdata[0]
                    if (iverbose > 1):
                        print(haloID, 'halo has taken subhalo main branc of ', branchfixSwapBranch,
                              'with progenitor of ',branchfixSwapBranchTail,
                              'having npart, stype and root tail of ', halodata[subTailSnap]['npart'][subTailIndex],
                              halodata[subTailSnap]['Structuretype'][subTailIndex], halodata[subTailSnap]['RootTail'][subTailIndex],
                        )
        else :
            branchfixSwapBranch = -2
    # if object is subhalo and host halo mergers with subhalo branch, fix host halo to take over
    # subhalo branch line
    else:
        haloHost = halodata[haloSnap]['hostHaloID'][haloIndex]
        haloHostIndex = np.uint64(haloHost % TEMPORALHALOIDVAL - 1)
        haloHostSnap = haloSnap
        haloHostRootHeadID = halodata[haloSnap]['RootHead'][haloHostIndex]
        if (iverbose > 1):
           print(haloID, 'subhalo has host ', haloHost,
              'that might take over subhalo main branch',
              'npart', halodata[haloSnap]['npart'][haloIndex], halodata[haloSnap]['npart'][haloHostIndex],
              'Stype', halodata[haloSnap]['Structuretype'][haloIndex], halodata[haloSnap]['Structuretype'][haloHostIndex]
              )

        if (iverbose > 2):
            mainbranchlength = sublifelength = 0
            curHalo = haloID
            curSnap = np.uint64(curHalo / TEMPORALHALOIDVAL)
            curIndex = np.uint64(curHalo % TEMPORALHALOIDVAL - 1)
            curRootTail = halodata[curSnap]['RootTail'][curIndex]
            curRootHead = halodata[curSnap]['RootHead'][curIndex]
            while (curRootTail == haloID):
                mainbranchlength += 1
                sublifelength += (halodata[curSnap]['hostHaloID'][curIndex] != -1)
                if (curHalo == curRootHead):
                    break
                curHalo = halodata[curSnap]['Head'][curIndex]
                curSnap = np.uint64(curHalo / TEMPORALHALOIDVAL)
                curIndex = np.uint64(curHalo % TEMPORALHALOIDVAL - 1)
                curRootTail = halodata[curSnap]['RootTail'][curIndex]
            print(haloID,'meets condition', (haloHostRootHeadID == haloRootHeadID), 'sublife',sublifelength,'mainbranch',mainbranchlength)

        if (haloHostRootHeadID == haloRootHeadID):
            haloHostHeadRank = treedata[haloSnap]['Rank'][haloHostIndex][0]
            haloHostHead = halodata[haloSnap]['Head'][haloHostIndex]
            haloHostHeadSnap = np.uint64(haloHostHead / TEMPORALHALOIDVAL)
            haloHostHeadIndex = np.uint64(haloHostHead % TEMPORALHALOIDVAL - 1)
            haloHostHeadRootTail = halodata[haloHostHeadSnap]['RootTail'][haloHostHeadIndex]
            # and this descendant is the primary descendant of the subhalo with no progenitor in question
            # have a fix.
            # #todo could add additional checks to see if subhalo becomes a halo later on
            # #todo might want to move foward until host is no longer main branch
            if (haloHostHeadRank > 0 and haloHostHeadRootTail == haloID):
                branchfixSwapBranch = haloHost
                branchfixSwapBranchTail = halodata[haloSnap]['Tail'][haloHostIndex]
        else :
            branchfixSwapBranch = -2
        if (iverbose > 1 and branchfixSwapBranch != -1):
            print(haloID, 'subhalo has lost main branch to', haloHost)

    return branchfixSwapBranch,branchfixSwapBranchTail


def FixBranchHaloSubhaloSwapBranchAdjustTree(numsnaps, treedata, halodata, numhalos,
                            nsnapsearch,
                            TEMPORALHALOIDVAL, iverbose,
                            haloID, haloSnap, haloIndex, haloRootHeadID,
                            branchfixSwapBranch, branchfixSwapBranchTail
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
        # now adjust, make descendant of subhalo the descendnat of the halo and update root tails
        # and progenitor of subhalo progenitor of halo
        # and if subhalo continues to exist but halo does not take over subhalo forward
        # branch line
        # first look at descendant of halo, see if it terminates and if so, take subhalo descendant
        oldHead = halodata[haloSnap]['Head'][haloIndex]
        oldHeadIndex = np.uint64(oldHead % TEMPORALHALOIDVAL-1)
        oldHeadSnap = np.uint64(oldHead / TEMPORALHALOIDVAL)
        oldHeadTail = halodata[oldHeadSnap]['Tail'][oldHeadIndex]
        if (haloID != oldHeadTail) :
            halodata[haloSnap]['Head'][haloIndex] = branchfixHead
            halodata[haloSnap]['HeadSnap'][haloIndex] = branchfixHeadSnap
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
        # also need to adjust any secondary progenitors of this subhalo
        ncount=0
        nprog=halodata[branchfixSnap]['Num_progen'][branchfixIndex]
        if (nprog > 1):
            searchlist=range(np.int32(branchfixSnap-1),np.int32(searchrange-1),-1)
            for i in searchlist:
                wdata=np.where(halodata[i]['Head']==branchfixSwapBranch)[0]
                halodata[i]['Head'][wdata] = haloID
                halodata[i]['HeadSnap'][wdata] = haloSnap
                ncount+=wdata.size
                if (ncount==nprog):
                    break
    else :
        haloHead = halodata[haloSnap]['Head'][haloIndex]
        haloHeadSnap = np.uint64(haloHead / TEMPORALHALOIDVAL)
        haloHeadIndex = np.uint64(haloHead % TEMPORALHALOIDVAL-1)
        haloRootHead = halodata[haloSnap]['RootHead'][haloIndex]
        if (iverbose > 1):
            print('subhalo ', haloID, 'left stranded in tree',
                  'but now also not primary progenitor of ', haloHead,
                  'host halo ', branchfixSwapBranch, 'takes over subhalo main branch',
                  'changing root tails from', haloID, 'to', branchfixRootTail,
                  'npartroottail', halodata[branchfixRootTailSnap]['npart'][branchfixRootTailIndex],
                  'styperoottail', halodata[branchfixRootTailSnap]['Structuretype'][branchfixRootTailIndex],
                  )

        # have halo point to subhalo descedant and descendant point to halo
        halodata[branchfixSnap]['Head'][branchfixIndex] = haloHead
        halodata[branchfixSnap]['HeadSnap'][branchfixIndex] = haloHeadSnap
        halodata[haloHeadSnap]['Tail'][haloHeadIndex] = branchfixSwapBranch
        halodata[haloHeadSnap]['TailSnap'][haloHeadIndex] = branchfixSnap
        # adjust any secondary progenitors of this subhalo to point to halo
        ncount=0
        nprog=halodata[haloSnap]['Num_progen'][haloIndex]
        if (nprog > 1):
            searchlist=range(np.int32(branchfixSnap-1),np.int32(searchrange-1),-1)
            for i in searchlist:
                wdata=np.where(halodata[i]['Head']==haloSnap)[0]
                halodata[i]['Head'][wdata] = branchfixSwapBranch
                halodata[i]['HeadSnap'][wdata] = branchfixSnap
                ncount+=wdata.size
                if (ncount==nprog):
                    break
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
    start = time.clock()
    start0 = time.clock()
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
    print(time.clock()-start0)
    # store number of fixes
    fixkeylist = ['TotalOutliers', 'HaloOutliers', 'SubOutliers',
                  'AfterFixTotalOutliers', 'AfterFixHaloOutliers', 'AfterFixSubOutliers',
                  'TotalFix', 'MergeFix', 'MergeFixBranchSwap', 'HaloSwapFix', 'SubSwapFix',
                  'NoFixAll', 'NoFixMerge', 'NoFixMergeBranchSwap', 'NoFixHaloSwap', 'NoFixSubSwap',
                  'NoMergeCandiate', 'Spurious']
    nfix = dict()
    for key in fixkeylist:
        nfix[key]= np.zeros(numsnaps)

    start1=time.clock()

    temparray = {'RootHead':np.array([], dtype=np.int64), 'ID': np.array([], dtype=np.int64), 'npart': np.array([], dtype=np.int32),
                'Descen': np.array([], dtype=np.int64), 'Rank': np.array([], dtype=np.int32), 'Merit': np.array([], np.float32),
                }
    secondaryProgenList = {'RootHead':np.array([], dtype=np.int64), 'ID': np.array([], dtype=np.int64), 'npart': np.array([], dtype=np.int32),
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
        wdata = np.where((treedata[isearch]['Num_descen']>0)*(halodata[isearch]['npart']>=npartlim_secondary))[0]
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
        secondaryProgenList['ID'] = np.concatenate([secondaryProgenList['ID'],halodata[isearch]['ID'][wdata]])
        secondaryProgenList['npart'] = np.concatenate([secondaryProgenList['npart'],halodata[isearch]['npart'][wdata]])
        if ('_Offsets' in treedata[isearch].keys()):
            temptemparray = treedata[isearch]['_Offsets'][wdata]
            secondaryProgenList['Descen'] = np.concatenate([secondaryProgenList['Descen'],treedata[isearch]['_Descens'][temptemparray]])
            secondaryProgenList['Rank'] = np.concatenate([secondaryProgenList['Rank'],treedata[isearch]['_Ranks'][temptemparray]])
            secondaryProgenList['Merit'] = np.concatenate([secondaryProgenList['Merit'],treedata[isearch]['_Merits'][temptemparray]])
        else:
            temptemparray=np.zeros(wdata.size, dtype=np.int64)
            for iw in range(wdata.size):
                temptemparray[iw]=treedata[isearch]['Descen'][wdata[iw]][0]
            secondaryProgenList['Descen'] = np.concatenate([secondaryProgenList['Descen'],temptemparray])
            temptemparray=np.zeros(wdata.size, dtype=np.int32)
            for iw in range(wdata.size):
                temptemparray[iw]=treedata[isearch]['Rank'][wdata[iw]][0]
            secondaryProgenList['Rank'] = np.concatenate([secondaryProgenList['Rank'],temptemparray])
            temptemparray=np.zeros(wdata.size, dtype=np.float32)
            for iw in range(wdata.size):
                temptemparray[iw]=treedata[isearch]['Merit'][wdata[iw]][0]
            secondaryProgenList['Merit'] = np.concatenate([secondaryProgenList['Merit'],temptemparray])
    print('Finished building temporary array for quick search containing ',num_with_more_descen)
    print('Finished building temporary secondary progenitor array for quick search containing ',num_secondary_progen)
    print('in',time.clock()-start1)

    #find all possible matches to objects with no primary progenitor
    start1 = time.clock()
    noprogID = np.array([],dtype=np.int64)
    noprogRootHead = np.array([],dtype=np.int64)
    #noprognpart = np.array([],dtype=np.int64)
    for i in range(numsnaps):
        noprog = np.where((halodata[i]['Tail'] == halodata[i]['ID'])*(
            halodata[i]['npart'] >= npartlim)*(halodata[i]['Head'] != halodata[i]['ID']))[0]
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
    print('Finished initalization in ',time.clock()-start1)
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
        branchfixSwapHaloOrSubhalo = branchfixSwapHaloOrSubhaloTail = -1

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
        if (mergeHalo == -1 and iswaphalosubhaloflag == 1):
            nfix['NoMergeCandiate'][haloSnap] += 1
            branchfixSwapHaloOrSubhalo, branchfixSwapHaloOrSubhaloTail = FixBranchHaloSubhaloSwapBranch(numsnaps, treedata, halodata, numhalos,
                        npartlim, meritlim,
                        TEMPORALHALOIDVAL, iverbose,
                        haloID, haloSnap, haloIndex, haloRootHeadID, mergeHalo
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
                        branchfixSwapHaloOrSubhalo, branchfixSwapHaloOrSubhaloTail
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
            branchfixSwapHaloOrSubhalo, branchfixSwapHaloOrSubhaloTail = FixBranchHaloSubhaloSwapBranch(numsnaps, treedata, halodata, numhalos,
                        npartlim, meritlim,
                        TEMPORALHALOIDVAL, iverbose,
                        haloID, haloSnap, haloIndex, haloRootHeadID, mergeHalo
                        )
        if (branchfixMerge < 0 and branchfixMergeSwapBranch < 0 and
            branchfixMergeSwapBranch < 0 and
            branchfixSwapHaloOrSubhalo < 0):
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
        else:
            if (branchfixSwapHaloOrSubhalo > -1):
                if (halodata[haloSnap]['hostHaloID'][haloIndex] == -1):
                    nfix['HaloSwapFix'][haloSnap] += 1
                else :
                    nfix['SubSwapFix'][haloSnap] += 1
                FixBranchHaloSubhaloSwapBranchAdjustTree(numsnaps, treedata, halodata, numhalos,
                        nsnapsearch,
                        TEMPORALHALOIDVAL, iverbose,
                        haloID, haloSnap, haloIndex, haloRootHeadID,
                        branchfixSwapHaloOrSubhalo, branchfixSwapHaloOrSubhaloTail
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
    # if checking tree, make sure root heads and root tails match when head and tails indicate they should
    if (ichecktree):
        irebuildrootheadtail = False
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

    for i in range(numsnaps):
        noprog = np.where((halodata[i]['Tail'] == halodata[i]['ID'])*(
            halodata[i]['npart'] >= npartlim)*(halodata[i]['Head'] != halodata[i]['ID']))[0]
        nfix['AfterFixTotalOutliers'][i] = noprog.size
        nfix['AfterFixHaloOutliers'][i] = np.where(halodata[i]['hostHaloID'][noprog] == -1)[0].size
        nfix['AfterFixSubOutliers'][i] = nfix['AfterFixTotalOutliers'][i] - nfix['AfterFixHaloOutliers'][i]
    print('Done fixing branches', time.clock()-start)
    print('For', np.sum(numhalos), 'across cosmic time')
    print('Corrections are:')
    for key in fixkeylist:
        print(key, np.sum(nfix[key]))
    if (iverbose > 1):
        print('With snapshot break down of ')
        for i in range(numsnaps):
            print('snap', i, 'with', numhalos[i], 'Fixes:')
            print([[key, nfix[key][i]] for key in fixkeylist])
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

    start = time.clock()
    start0 = time.clock()
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
    start1=time.clock()
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
    print('in',time.clock()-start1)

    #find all objects with no primary progenitor
    start1 = time.clock()
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

    print('Done adjusting secondaries ', time.clock()-start1)
    print('For', np.sum(numhalos), 'across cosmic time')
    print('Corrections are:')
    for key in fixkeylist:
        print(key, np.sum(nfix[key]))

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
            "Particle_Types", data=np.array(cattemp, dtype=np.int64))
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
                "Particle_Types", data=np.array(cattemp, dtype=np.int64))
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
