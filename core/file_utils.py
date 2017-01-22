import urllib2
import gzip
import zipfile
import errno
import os


def ensureDirectoryExists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def download(urls, destination):

    files = []

    try:
        for url in urls:
            print 'Downloading', url
            response = urllib2.urlopen(url)
            
            filename = os.path.join( destination, os.path.basename(url) )
            with open(filename, "wb") as local_file:
                local_file.write(response.read())
                files.append( filename )

        print "Done"

    except urllib2.URLError as e:
        print "Failed to download:", e.reason

    return files


def unzip(files, destination):

    for filename in files:
        print 'Unzipping', filename

        # zip_ref = zipfile.ZipFile( filename, 'r' )
        # zip_ref.extractall( destination )
        # zip_ref.close()

        with gzip.open(filename, 'rb') as in_file:
            with open(os.path.splitext(filename)[0], 'wb') as out_file:
                out_file.write( in_file.read() )


def ensureDownloadDirectoryExists(scriptPath):
    '''Creates a directory for downloads in the same directory as the script'''

    this_dir = os.path.dirname( os.path.abspath(scriptPath) )
    path = os.path.join( this_dir, 'download' )

        
    ensureDirectoryExists(path)

    return path


def removeFiles(files):

    for filename in files:
        os.remove(filename)


