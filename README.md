# cc_mpi.py
MPI4PY Cross-correlation Code for Pre-Downloaded Waveform file 

- Currently set-up for conversion of waveforms to hypoDD input files for xcorr
- Edits can be made for a more universal cross-correlation method -- email katherine.biegel@ucalgary.ca


For this script mpi4py is needed.
Obspy is also required.


To run execute using:
mpirun -np [number of processes] python cc_mpi.py


The following lines need to be edited to run correctly:

line 122:  nch is the integer value of the number of data chunks to be sent to workers 
  - this should be larger than the number of workers but not too large as to increase communication overhead

line 129: files is the string location of the waveform files to be cross-correlated
  - this can be a directory location or a subset of a directory

lines 138-145: 
  - client and inventory are both specified depending on the needed obspy information although this can be edited to read in files as well
  - eventnames is a list of str event tags as recorded in IRIS or other catalogs
           - for large examples this can be edited to read in eventnames from file which will likely be more efficient


OUTPUTS:
cc_arr.npy          ---- non-sorted cross-correlation result array
cc_arr_sorted.npy   ---- sorted cross-correlation result array

  - Arrays are stored in the form [:,0] = Event 1 ID
                                  [:,1] = Event 2 ID
                                  [:,2] = Station ID
                                  [:,3] = P Cross-correlation dt time
                                  [:,4] = P Cross-correlation max amplitude
                                  [:,5] = S Cross-correlation dt time
                                  [:,6] = S Cross-correlation max amplitude

MPI_CCS.cc          ---- output in the form of a hypoDD cc data file
