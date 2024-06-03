#!/usr/bin/env python
# coding: utf-8
"""
----------------------------------------------------------------------------
pdrinex.py  Read data from RINEX observation and navigation files into Pandas dataframes.

Created by: Hans van der Marel
Date:       12 April 2021
Modified:   -

Copyright: Hans van der Marel, Delft University of Technology, 2021
Email:     h.vandermarel@tudelft.nl
Github:    -
----------------------------------------------------------------------------
Functions:

Read RINEX observation and navigation files:

    rnxReadObs(rnxFilename: str, 
	           verbose: int = 0) -> pd.DataFrame
        Read RINEX observation data from file.
    
	rnxReadNav(rnxFilename: str, 
               verbose: int = 0) -> pd.DataFrame
        Read RINEX navigation data from file.          
    
High level data retrieval and information      
    
    rnxSelSat(rnxDf: pd.DataFrame, 
              sat: str, 
              verbose: int = 0) -> pd.DataFrame :
	    Retrieve RINEX observation data for a single satellite as pandas dataframe.

    rnxSelSys(rnxDf: pd.DataFrame,
              sys: str,
              verbose: int = 0) -> pd.DataFrame :
	    Retrieve RINEX observation data for a single system as pandas dataframe.

    rnxPrintMetadata(rnxDf: pd.DataFrame)
        Print meta data information stored in rnxDf.attrs and derived from the RINEX header
        and file parsing. Meta data itself can be accesssed through rnxDf.attrs['key]'.
        
Frequencies and units

    rnxGetFreqs(obstypes: list or pd.dataframe, 
                satId: str, 
                fcns: dict  = None, 
                wavelength: bool = False) -> dict:
        Get frequency or wavelength for list of observation types.    
    
    rnxGetUnits(obstypes: list or pd.dataframe) -> dict:
        Get measurement units for list of observation types.
           
Low level information retrieval and validation 
    
	rnxVersion(line: str or dict) -> (float, str, str):
        Check the first line of a RINEX/CRINEX file and return the version number and file type.
    
	rnxObsTypes(rnxHeader: dict, 
                verbose: int = 0) -> ( dict, int ):
        Retrieve RINEX observation types and maximum number of types from RINEX file header data.
    
	rnxGetInfo(rnxHeader: dict) -> dict:
        Parse RINEX header for version, interval, position, antenna offsets, marker, antenna and receiver names.

	rnxCheckSatids(satIds: list or np.array, 
	               repair: bool = False, 
				   verbose: int  = 0) -> dict:
    
        Check satellite identifiers from RINEX observation file and (optionally, default)
		raise an error when there are invalid satellite identifiers. 

    rnxCheckTimestamps(epochTimestamps: list or np.array, 
	                    rnxInfo: dict = None, verbose : int  = 0) -> dict:
    
        Check timestamps from RINEX observation file, raise an error
		when there are duplicate time stamps or when timestamps are not strictly
		ascending, and return information on the interval, first and last 
		epoch, gaps, etc
	
	rnxGetSatIds(rnxDf: pd.DataFrame, 
                 verbose: int = 0) -> np.ndarray :
        Retrieve a list of three character satellite identifiers from a pandas RINEX observation dataframe.

    rnxGetTimestamps(rnxDf: pd.DataFrame, 
                 verbose: int = 0) -> np.ndarray :   
        Retrieve epoch timestamps (or epoch numbers) from a pandas RINEX observation dataframe.
        
    rnxGetFcnsFromHeader(rnxHeader: dict or pd.DataFrame) -> dict:
        Get Glonass frequency numbers from RINEX observation headers.
    
    rnxGetFcnsFromNav(rnxNav: pd.DataFrame) -> dict:
        Get Glonass frequency numbers from RINEX navigation DataFrame.
    
Examples:

    import pdrinex as rnx


(c) Hans van der Marel, Delft University of Technology, 2021.
"""
 
import pandas as pd
import numpy as np
from io import StringIO

def rnxReadObs(rnxFilename: str, 
               verbose: int = 0) -> ( pd.DataFrame, dict, list ):
    """
    Read RINEX observation data from file.
    
    Parameters
    ----------
    rnxFilename : str
        RINEX observation filename
    verbose : int, default=0
        Verbosity level (0 is no output, 1 some, 2 more)

    Returns
    -------
    rnxDf : pandas.DataFrame
        Pandas multi index dataframe with the RINEX observations. The RINEX meta data, timestamps and other
        data are stored as attributes in the data frame (rnxDf.attrs).
        
    Raises
    ------
    ValueError
        If the file is not a RINEX observation file 
        If the RINEX observation file is other than a version 3 file (version 2 files not yet supported)
        If the time stamps are not strictly ascending or when there are duplicates
        If there are invalid satellite identifiers

    Notes
    -----
    Meta data is stored as a dictionary in rnxDf.attrs (attributes). The meta data is derived from the
    RINEX header records, epoch data and other results. Common meta data attributes are 
    {
        'version' : float
        'rnxtype' : str {'OBS', 'NAV', 'CRX'}
        'systems' : str {'G', 'R', 'E', 'C', 'S', 'I', 'J'} or {'M'} for mixed
        'marker' : str 
        'rectype' : str
        'anttype' : str
        'obstypes' : dict with the observation types for each system
        'interval' : float
        'timerange' : ( datetime64[ns], datetime64[ns] )
        'timesystem' : str
        'position' : (float, float, float) approximate (!) position
        'antoffset' : (float, float, float) antenna offset in Height, East and North
        'epochTimestamps' : list[datetime64[ns]]
        'rnxHeader' : dict with the original RINEX header data
    } 
    The meta data can be printed with the function 'rnxPrintMetadata(rnxDf)'.
        
    Examples
    --------
    Simple example to read a RINEX observation file and print the meta data

    >>> rnxDf  = rnxReadObs('ZEGV00NLD_R_20201190000_01D_30S_MO.rnx')
    >>> rnxPrintMetadata()
    
    Another file with raised verbosity level (includes printing meta data) amongst other things

    >>> rnxDf = rnxReadObs('ZEGV00NLD_R_20201190000_01D_30S_MO.rnx', verbose=1)
    
    (c) Hans van der Marel, Delft University of Technology, 2021
    """ 

    with open(rnxFilename,'r') as rnxfile:
    
        # Read first line and check rinex version and file type (skip blank lines)

        maxBlankLines=5
        for numBlankLines in range(maxBlankLines):
            line = rnxfile.readline()
            if line.strip(): break

        if numBlankLines == maxBlankLines - 1 or not line:
            raise ValueError(f"Cannot find a valid header line in {rnxFilename}") 

        version, rnxtype, systems = rnxVersion(line)
        
        if verbose > 0: print('RINEX type and version:',rnxtype,version)
            
        if not rnxtype == 'OBS':
            raise ValueError(f"RINEX observation file expected, this is a RINEX {rnxtype} file.") 
            
        if version < 3:
            raise ValueError(f"RINEX file version {version} is not (yet) supported, must be RINEX version 3.") 

        # Initialize rnxHeader dictionary and add the header line to the dict
        
        rnxHeader = {}
        headerLabel = line[60:].strip()       
        rnxHeader[headerLabel] = line[0:60]  

        # Read rinex header block and add lines to rnxHeader

        if verbose > 1: print(line.rstrip())

        for line in rnxfile:
            if "END OF HEADER" in line:
                break
        
            if verbose > 1: print(line.rstrip())
        
            headerLabel = line[60:].strip()        # Rinex header label
            if headerLabel not in rnxHeader:       # Store string in rnxHeader
                rnxHeader[headerLabel] = line[0:60]  
            else:                                  # Append to existing rnxHeader 
                tmp = rnxHeader[headerLabel]
                if not isinstance(tmp, list):
                    tmp = [ tmp ]
                tmp.append( line[0:60] )
                rnxHeader[headerLabel] = tmp

        if verbose > 0: print('Header fields:',rnxHeader.keys())
    
        # Process observation types
    
        (obsTypes, maxTypes) = rnxObsTypes(rnxHeader, verbose=verbose) 
		
        # Prepare colspecs and column names for pd.read_fwf
    
        colspecs = [ (0,3) ]
        colnames = [ 'SatId' ]
        for k in range(maxTypes):
            colspecs.append( (3+k*16,3+k*16+14))
            colnames.append(k)

        if verbose > 1: print(colspecs)
    
        # Process epoch blocks 

        epochTimestamps = []
        epochFlags = []
        epochNumSats = []
        epochDfs = []
        events = []

        for line in rnxfile:
        
            # Process epoch header
        
            if not line.startswith('>'):           # this is unexpected, 
                break

            if verbose > 1: print(line.rstrip())
       
            epochTime = line[2:30]
            epochFlag = line[31:32]
            numSat = int(line[32:35])

            # handle special events 
        
            if int(epochFlag) > 1:                # special events, skip (for the time being)
                event={}
                event['epochTime'] = epochTime
                event['epochFlag'] = epochFlag
                event['eventData'] = []
                for _ in range(numSat):
                    line = rnxfile.readline()
                    event['eventData'].append(line.rstrip())

                events.append(event)
                continue
        
            # Convert formatted time to pandas timestamp
        
            epochTimestamp = pd.Timestamp( year=int(epochTime[0:4]), month=int(epochTime[5:7]), day=int(epochTime[8:10]),
                   hour=int(epochTime[11:13]), minute=int(epochTime[14:16]), second=int(epochTime[17:19]),
                  microsecond=int(epochTime[20:26]), nanosecond=int(epochTime[26:27]) * 1000 )

            # Add epoch data to lists

            epochTimestamps.append(epochTimestamp)
            epochFlags.append(epochFlag)
            epochNumSats.append(numSat)
 
            # Read epoch data block and save to epoch dataframe list

            df = pd.read_fwf(rnxfile, header=None, nrows=numSat, colspecs=colspecs, index_col=0, names=colnames)
            epochDfs.append(df)

        if verbose > 0: print('done reading rinex observation file')

    # Concatenate epochs in epochDfs to panda data frame

    # rnxDf = pd.concat(epochDfs, keys=range(len(epochTimestamps)), names=['epoch'])
    rnxDf = pd.concat(epochDfs, keys=epochTimestamps, names=['TimeStamp'])
	
	# Check satellite identifiers, if necessary and possible, translate bad names

    satids = rnxGetSatids(rnxDf)
    translate = rnxCheckSatids(satids, repair=True)
    if translate:
        print('reindexing with corrected satIds not yet implemented')

	# Dictionary with RINEX meta data, check timestamps and satellites

    # Get meta data from the RINEX HEADER 
    rnxMetadata = rnxGetInfo(rnxHeader)
    rnxMetadata['obsTypes'] = obsTypes
    rnxMetadata['maxTypes'] = maxTypes
    # Check timestamps for duplicates and ascending order, update metadata from header, and add new meta data
    rnxMetadata = rnxCheckTimestamps(epochTimestamps, rnxMetadata)
    rnxMetadata['epochTimestamps'] = epochTimestamps
    rnxMetadata['epochFlags'] = epochFlags
    rnxMetadata['epochNumSats'] = epochNumSats
    rnxMetadata['events'] = events
	# Add satellite identifiers
    rnxMetadata['satIds'] = rnxGetSatids(rnxDf)
    # Add RINEX header and filename to the meta data
    rnxMetadata['rnxHeader'] = rnxHeader
    rnxMetadata['rnxFilename'] = rnxFilename

    if verbose > 1: 
        rnxPrintMetadata(rnxMetadata)
	
	# Store rnxMetadata as attrs in rnxDf (experimental feature of dataframes)
    
    rnxDf.attrs = rnxMetadata

    if verbose > 0: print('pandas observation data frame ready')
            
    return rnxDf


def rnxVersion(line: str or dict) -> (float, str, str):
    """
    Check the first line of a RINEX/CRINEX file and return the version number and file type.
    
    Parameters
    ----------
    line : str or dict
       first line of RINEX/CRINEX file or dict with rnxHeader lines 

    Results
    -------
    version : float
        RINEX file version
    rnxtype : str {'OBS', 'NAV', 'CRX'}
        RINEX file type 
    systems : str 
        System or mixed systems ('M')
        
    Raises
    ------
    TypeError
        If input arguments are of the incorrect type
    ValueError
        If the RINEX version line does not has the proper format

    Examples
    --------
    >>> version, rnxtype, systems = rnxVersion(rnxHeader)
    >>> version, rnxtype, systems = 
    ...     rnxVersion('     3.03           OBSERVATION DATA    M (MIXED)           RINEX VERSION / TYPE')

    (c) Hans van der Marel, Delft University of Technology, 2021.
    """

    if isinstance(line,dict):
        line = line['RINEX VERSION / TYPE'] + 'RINEX VERSION / TYPE'
    elif not isinstance(line, str):
        raise TypeError('rnxVersion expects a string or dict as input argument')

    if len(line) < 80 or line[60:80] not in ('RINEX VERSION / TYPE', 'CRINEX VERS   / TYPE'):
        raise ValueError('The version line of the RINEX file header is corrupted.')

    try:
        version = float(line[:9])
    except ValueError:
        raise ValueError(f'Could not determine file version from {line[:9]}')

    if line[20:40] == 'COMPACT RINEX FORMAT' or line[60:80] == 'CRINEX VERS   / TYPE':
        rnxtype = 'CRX'
        systems = None
    elif line[20] == 'O' or 'OBS' in line[20:40]:
        rnxtype = 'OBS'   
        systems = line[40]
    elif line[20] == 'N' or 'NAV' in line[20:40]:
        rnxtype = 'NAV'  
        systems = line[4]
    else:
        raise ValueError(f'Could not determine file type from {line[20:40]}')
        
    return version, rnxtype, systems 


def rnxObsTypes(rnxHeader: dict, 
                verbose: int = 0) -> ( dict, int ):
    """
    Retrieve RINEX observation types and maximum number of types from RINEX file header data.
    
    Parameters
    ----------   
    rnxHeader: dict
        Dictionary with RINEX header fields
    verbose: int
        Verbosity level (0 is no output, 1 some, 2 more)

    Returns
    -------
    obsTypes: dict
        Dictionary with the observation types for each GNSS in the RINEX header
    maxTypes: int
        Maximum number of observation types in the RINEX file (need this for reading) 
        
    Examples
    --------
    >>> obsTypes, maxTypes = rnxObsTypes(rnxHeader, verbose=verbose) 
    
    (c) Hans van der Marel, Delft University of Technology, 2021
    """
    
    obsTypes = {}
    maxTypes = 0 
    if not isinstance(rnxHeader['SYS / # / OBS TYPES'], list):
        rnxHeader['SYS / # / OBS TYPES'] = [ rnxHeader['SYS / # / OBS TYPES'] ]
    if (verbose > 1): print(rnxHeader['SYS / # / OBS TYPES'])
    
    for line in rnxHeader['SYS / # / OBS TYPES']:
        if not line.startswith(' '):
            # read system, number of observation types, and observation types
            sys = line[0]
            numTypes = int(line[3:6])
            obsTypes[sys] = line[6:60].split()
            maxTypes = max(numTypes, maxTypes)
            remainingTypes = numTypes - 13
        else:
            # if more than 13 observation types, continue reading on next line
            obsTypes[sys] += line[6:60].split()
            remainingTypes -= 13

        if remainingTypes < 1: assert len(obsTypes[sys]) == numTypes

    if verbose > 0:
        print('Observation types:')
        for sys in obsTypes.keys():
            print(sys,obsTypes[sys])
        print('Maximum number of observation types',maxTypes)

    return (obsTypes, maxTypes)


def rnxGetInfo(rnxHeader: dict) -> dict:
    """
    Parse RINEX header for version, interval, position, antenna offsets, marker, antenna and receiver names.
    
    Parameters
    ----------
    rnxHeader : dict
        dict with rnxHeader lines 

    Results
    -------
    info : dict 
        dictionary, with
        info = {
            'version' : float
            'rnxtype' : str {'OBS', 'NAV', 'CRX'}
            'systems' : str {'G', 'R', 'E', 'C', 'S', 'I', 'J'} or {'M'} for mixed
            'marker' : str
            'rectype' : str
            'anttype' : str
            'interval' : float
            'timerange' : ( datetime64[ns], datetime64[ns] )
            'timesystem' : str
            'position' : (float, float, float)
            'antoffset' : (float, float, float) antenna offset in Height, East and North
        } 

    (c) Hans van der Marel, Delft University of Technology, 2021.
    """

    version, rnxtype, systems = rnxVersion(rnxHeader)
    
    # Marker name and number (combine these entries to a single name)
    
    try:
        marker = rnxHeader['MARKER NAME']
        marker = marker.strip()
        try:
            markernumber = rnxHeader['MARKER NUMBER']
            marker = marker + ' (' + markernumber.strip() + ')'
        except (KeyError, ValueError):
            pass
    except (KeyError, ValueError):
        try:
            markernumber = rnxHeader['MARKER NUMBER']
            marker = markernumber.strip()
        except (KeyError, ValueError):                               
            marker = None  

    # Receiver type
    
    try:
        rectype = rnxHeader['REC # / TYPE / VERS'][20:40]
        rectype = rectype.strip()
    except (KeyError, ValueError):
        rectype = None  
        
    # Antenna type

    try:
        anttype = rnxHeader['ANT # / TYPE'][20:40]
        anttype = anttype.strip()
    except (KeyError, ValueError):
        anttype = None  

    # Approximate position
        
    try:
        position = tuple([float(j) for j in rnxHeader['APPROX POSITION XYZ'].split()])
    except (KeyError, ValueError):
        position = ( None, None, None )  

    # Antenna offsets in Height, East and North direction
    
    try:
        antoffset = tuple([float(j) for j in rnxHeader['ANTENNA: DELTA H/E/N'].split()])
    except (KeyError, ValueError):
        antoffset = ( None, None, None )  

    # Data interval
    
    try:
        interval = float(rnxHeader['INTERVAL'][0:10])
    except (KeyError, ValueError):
        interval = None

    # Start and end times, time system

    try:
        line = rnxHeader['TIME OF FIRST OBS']
        timesystem = line[48:51].strip() 
        # Convert formatted time to pandas timestamp
        starttime = pd.Timestamp( year=int(line[0:6]), month=int(line[6:12]), day=int(line[12:18]),
            hour=int(line[18:24]), minute=int(line[24:30]), second=int(float(line[30:36])),
            microsecond=int(float(line[30:43]) % 1 * 1000000), nanosecond=int(float(line[30:43]) * 1000000 % 1 * 1000) )
    except (KeyError, ValueError):
        starttime = None
        timesystem = None

    try:
        line = rnxHeader['TIME OF LAST OBS']
        # Convert formatted time to pandas timestamp
        endtime = pd.Timestamp( year=int(line[0:6]), month=int(line[6:12]), day=int(line[12:18]),
            hour=int(line[18:24]), minute=int(line[24:30]), second=int(float(line[30:36])),
            microsecond=int(float(line[30:43]) % 1 * 1000000), nanosecond=int(float(line[30:43]) * 1000000 % 1 * 1000) )
    except (KeyError, ValueError):
        endtime = None
     
    # Return dictionary with info

    info = {
        'version' : version,
        'rnxtype' : rnxtype,
        'systems' : systems,
        'marker' : marker,
        'rectype' : rectype,
        'anttype' : anttype,
        'interval' : interval,
        'timerange' : (starttime, endtime),
        'timesystem' : timesystem,
        'position' : position,
        'antoffset' : antoffset,
        }
                                    
    return info                                    


def rnxCheckTimestamps(epochTimestamps: list or np.array, rnxInfo: dict = None, verbose : int  = 0) -> dict:
    """
    Check timestamps from RINEX observation file, raise an error when there are duplicate time stamps
    or when timestamps are not strictly ascending, and return information on the interval, first and last
    epoch, gaps, etc.
    
    Parameters
    ----------
    epochTimeStamps : list[datetime64[ns]] or np.array[datetime64[ns]]  
        List or numpy array with the epoch times as pandas timestamps (datetime64[ns])      
    rnxInfo : dict , optional
        dict with meta data from RINEX header, obtained by rnxGetInfo(), if present the rnxInfo
        will be compared and updated with the new analysis results.
    verbose: int
        Verbosity level (0 is no output, 1 some, 2 more)

    Results
    -------
    info : dict 
        info = {
            'interval' : float [seconds] ,
            'timerange' : (datetime64[ns], datetime64[ns]) ,
            'isregular' : bool ,
            'hasgaps' : bool ,
            'maxgap' : float [seconds] ,
            'numepochs' : int ,
            'expected' : int ,
            'missing' : int ,
        } 
        If optional 'rnxInfo' is an argument, then the information of 'rnxInfo' is added
        to the output 'info'.

    Raises
    ------
    ValueError
        If epochTimeStamps have duplicates
        If epochTimeStamps are not stricly ascending

    Examples
    --------
    >>> info = rnxCheckTimestamps(epochTimestamps)
    >>> rnxinfo = rnxGetInfo(rnxHeader)
    >>> rnxinfo = rnxCheckTimestamps(epochTimestamps, rnxinfo)

    (c) Hans van der Marel, Delft University of Technology, 2021.
    """
   
    if isinstance(epochTimestamps, list):
        epochTimestamps=np.array(epochTimestamps)
    elif not isinstance(epochTimestamps, np.ndarray):
        raise TypeError('rnxCheckTimestamps expects a list or numpy array as first input argument')

    # Determine interval and check that the array is strictly ascending and has no duplicate epochs
    
    dt=np.diff(epochTimestamps)

    interval = np.median(dt).total_seconds()                       # Median interval
    mininterval = np.min(dt).total_seconds()                       # Minimum and maximum interval
    maxinterval = np.max(dt).total_seconds()                       
        
    if verbose > 0: print(interval, mininterval, maxinterval)
        
    madinterval = np.median( dt - np.median(dt) ).total_seconds()  # Median Absolute Deviation of interval
    
    if mininterval < 0:
        raise ValueError('epochTimestamps must be strictly ascending')

    if mininterval < max( [ madinterval * 5, interval *.1 ] ):
        raise ValueError('epochTimestamps has duplicate values')
        
    # Check if there are gaps and if the data is regularly samples
    
    isregular = 5 * madinterval < interval * 0.1
    hasgaps = maxinterval > 2 * interval - min([ interval *.1, 5 * madinterval ])
    if hasgaps:
        maxgap = maxinterval - interval                           # gap is max interval minus nominal interval
    else:
        maxgap = None
    
    # Start and end times
    
    starttime = epochTimestamps[0]
    endtime = epochTimestamps[-1]
    
    # Expected and missing number of epochs
    
    numepochs = len(epochTimestamps)
    if isregular:
        expected = round( ( endtime - starttime ).total_seconds() / interval ) + 1
        missing = expected - numepochs
    else:
        expected = None
        missing = None

    # Return dictionary with info

    info = {
        'interval' : interval,
        'timerange' : (starttime, endtime),
        'isregular' : isregular,
        'hasgaps' : hasgaps,
        'maxgap' : maxgap,
        'numepochs' : numepochs,
        'expected' : expected,
        'missing' : missing,
    }
    
    # Optionally add fields from rnxInfo, after comparing ... 

    if not rnxInfo == None:
        
        for key in info:
            if key in rnxInfo:         # Compare
                if key == 'interval' :
                    if rnxInfo[key] != None and info[key] != rnxInfo[key] :
                        print('Interval does not match: Timestamps',info[key],' vs RINEX header',rnxInfo[key])
                elif key == 'timerange' :
                    if rnxInfo[key][0] != None and info[key][0] != rnxInfo[key][0] :
                        print('Starttime does not match: Timestamps',info[key][0],' vs RINEX header',rnxInfo[key][0])
                    if rnxInfo[key][1] != None and info[key][1] != rnxInfo[key][1] :
                        print('Endtime does not match: Timestamps',info[key][1],' vs RINEX header',rnxInfo[key][1])
            
            rnxInfo[key] = info[key]  # Add 

        info = rnxInfo

    return info


def rnxCheckSatids(satIds: list or np.array, repair: bool = False, verbose: int  = 0) -> dict:
    """
    Check satellite identifiers from RINEX observation file and (optionally, default) raise an error 
    when there are invalid satellite identifiers. 
    
    Parameters
    ----------
    satIds : list[str] or np.array[str]  
        List or numpy array with the satellite identifiers as strings      
    repair : bool, default = False
        If True, don't raise an error, but output a dict with repairs
    verbose: int
        Verbosity level (0 is no output, 1 some, 2 more)

    Results
    -------
    translate : dict or None 
        translate = {
            'old SatId' : 'new SatId' ,
            ...
        } 
        If no repairs need to be done returns None

    Raises
    ------
    TypeError
        If satIds is not a list or numpy array
    ValueError
        If satIds are not three characters or contain an invalid system identifier
        If satIds contains blanks (only if 'repair' == False)

    Examples
    --------
    >>> translate = rnxCheckSatids(satIds)

    (c) Hans van der Marel, Delft University of Technology, 2021.
    """
   
    if isinstance(satIds, np.ndarray):
        satIds = satIds.tolist()
    elif not isinstance(satIds, list):
        raise TypeError('rnxCheckTimestamps expects a list or numpy array as first input argument')

      
    permitted = 'GRSECJI'

    badIds = {}
    translate = {}
    
    for x in satIds:
        if len(x) != 3:
            badIds[x] = f'Wrong length ({len(x)})'
        elif x[0] not in permitted:
            badIds[x] = f'Invalid system {x[0]}'
        elif not x[1].isnumeric():
            if x[1] == ' ' and repair:
                translate[x] = x.replace(' ','0')
            else:
                badIds[x] = 'First digit not numeric'
        elif not x[2].isnumeric():
            badIds[x] = 'Last digit not numeric'
            
    if verbose > 0:
        if len(badIds) > 0: print('Bad satIds:',badIds)
        if len(translate) > 0: print('Translatable satIds:',translate)          
        
    if len(badIds) > 0:
        raise ValueError('There are bad satellite identifiers')

    if len(translate) == 0: 
        translate = None
        
    return translate


def rnxGetSatids(rnxDf: pd.DataFrame, 
                 verbose: int = 0) -> np.ndarray :
    """
    Retrieve a list of three character satellite identifiers from a pandas RINEX observation dataframe.
    
    Parameters
    ----------
    rnxDf: pandas.DataFrame
        Pandas multi index dataframe with the RINEX observations
    verbose: int
        Verbosity level (0 is no output, 1 some, 2 more)

    Returns
    -------
    satIds: numpy.ndarray[str]
        List with three character satellite identifiers (e.g. G05)
        
    Examples
    --------
    >>> satIds = rnxGetSatIds(rnxDf) 
    
    (c) Hans van der Marel, Delft University of Technology, 2021
    """

    satIds = rnxDf.index.get_level_values('SatId').unique()
    satIds = np.sort(satIds)
    
    if verbose > 0: 
        print(satIds)
        print('Number of satelites:',len(satIds))
        
    return satIds


def rnxGetTimestamps(rnxDf: pd.DataFrame, 
                     verbose: int = 0) -> np.ndarray :
    """
    Retrieve epoch timestamps (or epoch numbers) from a pandas RINEX observation dataframe.
    
    Parameters
    ----------
    rnxDf: pandas.DataFrame
        Pandas multi index dataframe with the RINEX observations
    verbose: int
        Verbosity level (0 is no output, 1 some, 2 more)

    Returns
    -------
    epochTimestamps: numpy.ndarray[datetime64[ns]] or numpy.ndarray[int]
        epoch timestamps as pandas timestamps or epoch numbers
        
    Examples
    --------
    >>> epochTimestamps = rnxGetTimestamps(rnxDf) 
    
    (c) Hans van der Marel, Delft University of Technology, 2021
    """

    epochTimestamps = rnxDf.index.get_level_values('TimeStamp').unique()
    epochTimestamps = np.sort(epochTimestamps)    
    if verbose > 0: 
        print(epochTimestamps)
        print('Number of epochs:',len(epochTimestamps))
        
    return epochTimestamps


def rnxSelSat(rnxDf: pd.DataFrame, 
              satId: str, 
              verbose: int = 0) -> pd.DataFrame :
    """
    Retrieve RINEX observation data for a single satellite as pandas dataframe.
    
    Parameters
    ----------
    rnxDf: pandas.DataFrame
        Pandas multi index dataframe with the RINEX observations
    satId: string
        Three character satellite identifier (e.g. G05)
    verbose: int
        Verbosity level (0 is no output, 1 some, 2 more)

    Returns
    -------
    satDf: pandas.DataFrame
        Pandas dataframe with the RINEX observations for a single satellite, with Timestamps as index, and
        observation types as column names. Observations types without data will be dropped.
        
    Examples
    --------
    New data frame with the data for GPS G05 with observation types as column names
    
    >>> satDf = rnxSelSat(rnxDf, 'G05') 
    
    (c) Hans van der Marel, Delft University of Technology, 2021
    """

    # Subset with data for a single satellite
    
    satDf = rnxDf.loc[:,satId,:] 

    # Change the column names in satDf to the observation types

    obsTypes = rnxDf.attrs['obsTypes'][satId[:1]]

    tmpdict = dict(zip(satDf.columns,obsTypes))    
    satDf = satDf.rename(columns=tmpdict)
    satDf.columns.rename(name='ObsType', inplace=True)
    
    # Drop columns will all NaN's
    
    satDf =  satDf.dropna(axis=1, how='all')

    # Add satellite identifier as attribute (do not confuse with satIds, which is the global list of ids)

    if not satDf.attrs:
        satDf.attrs=rnxDf.attrs

    satDf.attrs['satId'] = satId

    return satDf


def rnxSelSys(rnxDf: pd.DataFrame, 
              sysId: str,
              unstack: bool = False,
              verbose: int = 0) -> pd.DataFrame :
    """
    Retrieve RINEX observation data for a single system as pandas dataframe.
    
    Parameters
    ----------
    rnxDf: pandas.DataFrame
        Pandas multi index dataframe with the RINEX observations
    sysId: str
        Single character system identifier (e.g. G)
    unstack: bool
        If True unstack the data and pair satellites and observation types into multi index
        columns. If False (default), columns are observation types, and rows are a multi index.
    verbose: int
        Verbosity level (0 is no output, 1 some, 2 more)

    Returns
    -------   
    sysDf: pandas.DataFrame
        Pandas dataframe with the RINEX observations for a single system, with Timestamps as index, and
        observation types as column names. The observations types and number of types, thus columns,
        depend on the system selected. 
        
    Examples
    --------
    Retrieve all data for the GPS system as a multi index pandas data frame
    
    >>> sysDf = rnxSelSys(rnxDf, 'G') 
    
    (c) Hans van der Marel, Delft University of Technology, 2021
    """

    # Subset with data for a single system

    sel = np.char.startswith(rnxDf.index.get_level_values('SatId').to_numpy(dtype=str),sysId)
    sysDf = rnxDf.loc(axis=0)[sel]

    # allSats=np.sort(rnxDf.index.get_level_values('sys').unique())
    # satDf = rnxDf.loc(axis=0)[:,allSats]
    
    # Change the column names in satDf to the observation types

    obsTypes = rnxDf.attrs['obsTypes'][sysId]

    tmpdict = dict(zip(sysDf.columns,obsTypes))    
    sysDf = sysDf.rename(columns=tmpdict)
    sysDf.columns.rename(name='ObsType', inplace=True)
    
    # Drop columns will all NaN's
    
    sysDf =  sysDf.dropna(axis=1, how='all')

    # Optionally unstack into multi index columns
    
    if unstack:
        sysDf = sysDf.unstack().reorder_levels([1, 0], axis=1).sort_index(axis=1).dropna(axis=1,how='all')
    
    if not sysDf.attrs:
        sysDf.attrs=rnxDf.attrs
    
    return sysDf

def rnxPrintMetadata(rnxDf: pd.DataFrame or dict):
    """
    Print RINEX meta data from a pandas dataframe or dict.
    
    Parameters
    ----------
    rnxDf: pandas.DataFrame or dict
        Pandas multi index dataframe with the RINEX observations or a dict
        with the meta data
    
    (c) Hans van der Marel, Delft University of Technology, 2021
    """

    if isinstance(rnxDf,pd.DataFrame):
        rnxMetadata = rnxDf.attrs
    elif isinstance(rnxDf,dict):
        rnxMetadata = rnxDf
    else:
        raise TypeError('Input argument must be a dict or pandas data frame')

    for attr in rnxMetadata:
        if isinstance(rnxMetadata[attr],list):
            print(attr,': ',np.array(rnxMetadata[attr]))
        elif attr == 'rnxHeader':
            print(attr,': ','<not printed>')
        elif isinstance(rnxMetadata[attr],dict):
            print(attr,': ')
            for key in rnxMetadata[attr]:
                print('   ',key,':',rnxMetadata[attr][key])
        else:
            print(attr,': ',rnxMetadata[attr])

    return


def rnxReadNav(rnxFilename: str, 
               verbose: int = 0) -> ( pd.DataFrame, dict ):
    """
    Read RINEX navigation data from file.
    
    Parameters
    ----------
    rnxFilename : str
        RINEX navigation filename
    verbose : int, default=0
        Verbosity level (0 is no output, 1 some, 2 more)

    Returns
    -------
    navDf : pandas.DataFrame
        Pandas dataframe with the RINEX navigation data
    rnxHeader : dict
        Dictionary with RINEX header fields and processed observation types (obsTypes, maxTypes) 
                
    Examples
    --------
    >>> navDf, navHeader = rnxReadNav('DLF100NLD_R_20171130000_01D_MN.rnx', verbose=1)
    
    (c) Hans van der Marel, Delft University of Technology, 2021
    """ 

    sysdef = dict(zip([ 'G', 'E', 'C', 'J', 'I', 'R', 'S'],[ 8, 8, 8, 8, 8, 4, 4]))
    
    with open(rnxFilename,'r') as rnxfile:
    
        # Read rinex header block

        rnxHeader = {}
    
        for line in rnxfile:
            if "END OF HEADER" in line:
                break
        
            if verbose > 1: print(line.rstrip())
        
            headerLabel = line[60:].strip()        # Rinex header label
            if headerLabel not in rnxHeader:       # Store string in rnxHeader
                rnxHeader[headerLabel] = line[0:60]  
            else:                                  # Append to existing rnxHeader 
                tmp = rnxHeader[headerLabel]
                if not isinstance(tmp, list):
                    tmp = [ tmp ]
                tmp.append( line[0:60] )
                rnxHeader[headerLabel] = tmp

        if verbose > 0: print('Header fields:',rnxHeader.keys())
        
        # Process navigation data blocks 

        navdata = []
        for line in rnxfile:

            if verbose > 2: print(line.rstrip())

            # Process epoch header
        
            sat=line[0:3].replace(' ','0')
            sys=line[0]
            if not sys in sysdef.keys():      # this is unexpected, 
                if verbose > 0: print('Junk: ',line.rstrip())
                continue

            date=line[4:14].replace(' ','-')
            time=line[15:23].replace(' ',':')
            record = sat + ' '+ date + ' ' + time + '{:57}'.format(line[23:].rstrip())
            for _ in range(1,sysdef[sys]):
                line = rnxfile.readline()
                record  = record + '{:76}'.format(line[4:].rstrip())

            if verbose > 1: print(record[:110] + ' ...')

            navdata.append(record.rstrip())
            

        if verbose > 0: print('done reading rinex navigation file')

        # Convert nagivation data to pandas dataframe
        
        shortNames,_,_ = rnxNavFields()
        colnames = [ 'sat', 'datetime' ] + shortNames['GECIJ'] 
        if verbose > 0: print(colnames)
        
        TXTBUFFER = StringIO( '\n'.join(navdata).replace('D', 'E') )
        navDf = pd.read_fwf(TXTBUFFER, header=None, lineterminator='\n', 
            widths=[ 3, 20, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 
                            19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 
                            19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19 ],
            parse_dates=[1],
            names=colnames)

        if verbose > 0: print('panda navigation dataframe ready')

    return (navDf, rnxHeader)


def rnxGetUnits(obstypes: list or pd.dataframe) -> dict:
    """
    Get measurement units for list of observation types.
    
    Parameters
    ----------
    obstypes: list or pandas.DataFrame
        List with observation types or pandas dataframe with the RINEX observations

    Returns
    -------
    obsunits: dict
        Dictionary obsunits['obstype'] -> str with measurement unit
        
    Examples
    --------
    Get a dict with the units
    
    >>> units = rnxGetUnits(obstypes) 

    See also
    --------
    rnxGetFreqs
        Retrieve frequencies or wavelengths for a list of RINEX observation types

    (c) Hans van der Marel, Delft University of Technology, 2021
    """

    if isinstance(obstypes,pd.DataFrame):
        obstypes = obstypes.columns.tolist()
    elif not isinstance(obstypes,list):
        raise  TypeError('must be list or pandas dataframe')
    
    defunits={
       'C' : '[m]',
       'L' : '[cycles]',
       'S' : '[dB-Hz]',
       'D' : '[Hz]',
    }

    obsunits={}

    for obstype in obstypes:
        if obstype[0] in defunits:
            obsunits[obstype] = defunits[obstype[0]]
        else:
            obsunits[obstype] = '[-]'
        
    return obsunits


def rnxGetFreqs(obstypes: list or pd.dataframe, 
                satId: str, 
                fcns: dict  = None, 
                wavelength: bool = False) -> dict:
    """
    Get frequency or wavelength for list of observation types.
    
    Parameters
    ----------
    obstypes: list or pandas.DataFrame
        List with observation types or pandas dataframe with the RINEX observations
    satId: str
        String with the system Id or satellite Id. For Glonass the satellite Id is required, for other systems
        the system Id is sufficient.
    fcns: dict, optional
        Dictionary with the frequency channel numbers for Glonass satellites
    wavelength: bool, optional
        If True, returns the wavelength instead of the frequency

    Returns
    -------
    obsunits: dict
        Dictionary obsfreq['obstype'] -> float with the frequency in [Hz], or wavelenght in [m]
        
    Notes
    -----
    The frequency is only defined for carrier-phase and Doppler observation types. For pseudo-range
    types the frequency is zero, for signa-to-noise density ratios a Nan is returned.
    
    Except for Glonass, the frequencies are the same for every satellite. For Glonass, the frequencies
    depend on the Glonass frequency channel number. The input of a dict with Glonass frequency channel
    numbers is mandatory if a list with Glonass observation types is given. If the input is a
    a DataFrame then the frequency channel number is derived from the RINEX header data. Frequency 
    channel numbers are also available from RINEX navigation files

    Examples
    --------
    Get a dict with the GPS frequencies for the observations in obstype
    
    >>> freqs = rnxGetFreq(obstypes,'G') 

    See also
    --------
    rnxGetUnits
        Retrieve units for RINEX observation types
    rnxGetFcnsFromHeader
        Retrieve Glonass frequency numbers from RINEX observation headers
    rnxGetFcnsFromNav
        Retrieve Glonass frequency numbers from RINEX navigation files

    (c) Hans van der Marel, Delft University of Technology, 2021
    """

    if isinstance(obstypes,pd.DataFrame):
        obstypes = obstypes.columns.tolist()
    elif not isinstance(obstypes,list):
        raise  TypeError('must be list or pandas dataframe')

    c = 299792458
    f0 = 10.23e6
    nan = np.nan

    """    
    GPS frequencies
         1   L1   1574.42
         2   L2   1227.60
         5   L5   1176.45      
    Glonass frequencies
         1   G1   1602+nfreq*9/16    nfreq=-7...+12
         2   G2   1246+nfreq*7/16
    Galileo frequencies
         1   E1   1575.42  (L1)
         5   E5a  1176.42  (L5)
         7   E5b  1207.140  
         8   E5   1191.795
         6   E6   1278.75
    SBAS frequencies
         1   L1   1574.42
         5   L5   1176.45
    Compass frequencies
         -   B1-2 1589.742  (E1)
         2   B1   1561.098  (E2)
         7   B2   1207.14   (E5b)
         6   B3   1268.52  
    QZSS frequencies
         1   L1   1574.42   (L1)
         2   L2   1227.60   (L2)
         5   L5   1176.45   (L5,E5a)
         6   LEX  1278.75   (E6)
    """

    #                      L1    L2    L3    L4    L5    L6    L7    L8     (RINEX nomenclature)
    deffreqs={
       'G' :  np.array([  154,  120,  nan,  nan,  115,  nan,  nan,  nan  ]) * f0 ,
       'S' :  np.array([  154,  nan,  nan,  nan,  115,  nan,  nan,  nan  ]) * f0 ,
       'E' :  np.array([  154,  nan,  nan,  nan,  115,  125,  118, 116.5 ]) * f0 ,
       'C' :  np.array([  nan, 1561.098, nan, nan, nan, 1268.52, 1207.14, nan ]) * 1e6 ,
       'J' :  np.array([  154,  120,  nan,  nan,  115,  125,  nan,  nan  ]) * f0 ,
       'I' :  np.array([  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan  ]) ,
    }

    if satId[0] in deffreqs:
        # CDMA systems, easy
        freq = deffreqs[satId[0]]
    elif satId[0] == 'R':
        # Glonass frequencies
        #   1   G1   1602+nfreq*9/16    nfreq=-7...+12
        #   2   G2   1246+nfreq*7/16
        if fcns:
            nfreq = fcns[satId]
        else:
            raise RuntimeError('For Glonass the fcns input parameter is mandatory.')

        freq=np.array([ 1602.0 + nfreq * 9/16, 1246.0 + nfreq * 7/16, nan, nan, nan, nan, nan, nan ]) * 1e6
    else:
        raise ValueError(f'Unsupported system {satId[0]}')

    obsfreq={}
    obswavelength={}

    for obstype in obstypes:
        if obstype[0] == 'L' or obstype[0] == 'D':
            obsfreq[obstype] = freq[int(obstype[1])-1]
            obswavelength[obstype] = c/freq[int(obstype[1])-1]
        elif obstype[0] == 'C':
            obsfreq[obstype] = 0
            obswavelength[obstype] = np.Inf
        else:
            obsfreq[obstype] = nan
            obswavelength[obstype] = nan
            
    if wavelength:
        obsfreq = obswavelength

    return obsfreq


def rnxGetFcnsFromHeader(rnxHeader: dict or pd.DataFrame) -> dict:
    """
    Get Glonass frequency numbers from RINEX observation headers.
    
    Parameters
    ----------
    rnxHeader: dict or pandas.DataFrame
        Dict with RINEX headers pandas dataframe with the RINEX observations

    Returns
    -------
    fcns: dict
        Dictionary with Glonass frequency channels numbers, with satellite Ids as keys.
        
    (c) Hans van der Marel, Delft University of Technology, 2021
    """

    if isinstance(rnxHeader,pd.DataFrame):
        rnxHeader = rnxHeader.attrs['rnxHeader']
    elif not isinstance(rnxHeader,dict):
        raise  TypeError('Input parameter must be dict or pandas dataframe')

    slots=rnxHeader['GLONASS SLOT / FRQ #']
    if slots:
        line = ' '.join(slots)
        numsat = int(line[0:3])
        fcn_list = line[3:].split()
        key_list = fcn_list[0::2]
        value_list = map(int, fcn_list[1::2])    
        if len(key_list) != numsat or len(key_list) != len(key_list): 
            raise ValueError('There is a format error in the GLONASS SLOT / FRQ #')
        fcns=dict(zip(key_list,value_list))
    else:
        raise KeyError('No GLONASS frequency number data available from RINEX observation file header')

    return fcns

def rnxGetFcnsFromNav(rnxNav: pd.DataFrame) -> dict:
    """
    Get Glonass frequency numbers from RINEX navigation DataFrame.
    
    Parameters
    ----------
    rnxNav: pandas.DataFrame
        Pandas dataframe with the RINEX navigation data

    Returns
    -------
    fcns: dict
        Dictionary with Glonass frequency channels numbers, with satellite Ids as keys.
        
    (c) Hans van der Marel, Delft University of Technology, 2021
    """

    if not isinstance(rnxNav,pd.DataFrame):
        raise  TypeError('Input parameter must be a pandas dataframe')

    raise RuntimeError('This function is not yet implemented')

    # Find Glonass satellites 
    # idx=orb.satid(:,1) == 'R';

    # Get frequency numbers
    #fcn=orb.eph(idx,13);

    # Create dict
    
    # f(satnum)=fcn;

    # Check that there are no changes in fcn during the period of the navigation data

    #if any(fcn-slot2fcn(satnum)')
    #    warning('Glonass Frequency channel assignments changed during this navigation file')

    #return fcns
    return


def rnxNavFields() -> dict:
    
    """
    numFields = {
        'G' : 31,
        'C' : 31,
        'R' : 15,
        'S' : 15,
        'J' : 31,
        'E' : 31,
        'I' : 31
    }
    """

    shortNames = {
    'GECIJ' : ['svbi', 'svdr', 'svdrr',
                  'aode', 'crs', 'deltan', 'm0',
                  'cuc', 'e', 'cus', 'sqa',
                  'toe', 'cic', 'omega0', 'cis',
                  'i0', 'crc', 'omega', 'omegadot',
                  'idot', 'misc1', 'week', 'misc2',
                  'svac', 'svhe', 'misc3', 'misc4',
                  'how', 'misc5', 'spare0', 'spare1'],
       'RS' : ['svbi', 'svdr', 'mft',
                  'x', 'vx', 'ax', 'health',
                  'y', 'vy', 'ay', 'misc1',
                  'z', 'vz', 'az', 'misc2']        
    }
    
    genNames = {
    'GECIJ' : ['SVclockBias', 'SVclockDrift', 'SVclockDriftRate',
                  'IODE', 'Crs', 'DeltaN', 'M0',
                  'Cuc', 'Eccentricity', 'Cus', 'sqrtA',
                  'Toe', 'Cic', 'Omega0', 'Cis',
                  'Io', 'Crc', 'omega', 'OmegaDot',
                  'IDOT', 'misc1', 'Week', 'misc2',
                  'SISA', 'health', 'misc3', 'misc4',
                  'TransTime', 'misc5', 'spare0', 'spare1'],
       'RS' : ['SVclockBias', 'SVrelFreqBias', 'MessageFrameTime',
                  'X', 'dX', 'dX2', 'health',
                  'Y', 'dY', 'dY2', 'misc1',
                  'Z', 'dZ', 'dZ2', 'misc2']
    }
    navFields = {
       'G' : ['SVclockBias', 'SVclockDrift', 'SVclockDriftRate',
                  'IODE', 'Crs', 'DeltaN', 'M0',
                  'Cuc', 'Eccentricity', 'Cus', 'sqrtA',
                  'Toe', 'Cic', 'Omega0', 'Cis',
                  'Io', 'Crc', 'omega', 'OmegaDot',
                  'IDOT', 'CodesL2', 'GPSWeek', 'L2Pflag',
                  'SVacc', 'health', 'TGD', 'IODC',
                  'TransTime', 'FitIntvl', 'spare0', 'spare1'],
        'R' : ['SVclockBias', 'SVrelFreqBias', 'MessageFrameTime',
                  'X', 'dX', 'dX2', 'health',
                  'Y', 'dY', 'dY2', 'FreqNum',
                  'Z', 'dZ', 'dZ2', 'AgeOpInfo'],
        'S' : ['SVclockBias', 'SVrelFreqBias', 'MessageFrameTime',
                  'X', 'dX', 'dX2', 'health',
                  'Y', 'dY', 'dY2', 'URA',
                  'Z', 'dZ', 'dZ2', 'IODN'],
        'C' : ['SVclockBias', 'SVclockDrift', 'SVclockDriftRate',
                  'AODE', 'Crs', 'DeltaN', 'M0',
                  'Cuc', 'Eccentricity', 'Cus', 'sqrtA',
                  'Toe', 'Cic', 'Omega0', 'Cis',
                  'Io', 'Crc', 'omega', 'OmegaDot',
                  'IDOT', 'spare0', 'BDTWeek', 'spare1',
                  'SVacc', 'SatH1', 'TGD1', 'TGD2',
                  'TransTime', 'AODC', 'spare2', 'spare3'],
        'J' : ['SVclockBias', 'SVclockDrift', 'SVclockDriftRate',
                  'IODE', 'Crs', 'DeltaN', 'M0',
                  'Cuc', 'Eccentricity', 'Cus', 'sqrtA',
                  'Toe', 'Cic', 'Omega0', 'Cis',
                  'Io', 'Crc', 'omega', 'OmegaDot',
                  'IDOT', 'CodesL2', 'GPSWeek', 'L2Pflag',
                  'SVacc', 'health', 'TGD', 'IODC',
                  'TransTime', 'FitIntvl', 'spare0', 'spare1'],
        'E' : ['SVclockBias', 'SVclockDrift', 'SVclockDriftRate',
                  'IODnav', 'Crs', 'DeltaN', 'M0',
                  'Cuc', 'Eccentricity', 'Cus', 'sqrtA',
                  'Toe', 'Cic', 'Omega0', 'Cis',
                  'Io', 'Crc', 'omega', 'OmegaDot',
                  'IDOT', 'DataSrc', 'GALWeek',
                  'spare0',
                  'SISA', 'health', 'BGDe5a', 'BGDe5b',
                  'TransTime',
                  'spare1', 'spare2', 'spare3'],
        'I' : ['SVclockBias', 'SVclockDrift', 'SVclockDriftRate',
                  'IODEC', 'Crs', 'DeltaN', 'M0',
                  'Cuc', 'Eccentricity', 'Cus', 'sqrtA',
                  'Toe', 'Cic', 'Omega0', 'Cis',
                  'Io', 'Crc', 'omega', 'OmegaDot',
                  'IDOT', 'spare0', 'BDTWeek', 'spare1',
                  'URA', 'health', 'TGD', 'spare2',
                  'TransTime',
                  'spare3', 'spare4', 'spare5']
    }
    
    return shortNames, genNames, navFields


if __name__ == "__main__":
    rnxDf = rnxReadObs('DLF100NLD_R_20171130000_01D_30S_MO.rnx', verbose=1)   

    rnxPrintMetadata(rnxDf)

    satDf =  rnxSelSat(rnxDf, 'G05')

    navDf, navHeader = rnxReadNav('DLF100NLD_R_20171130000_01D_MN.rnx', verbose=1)


