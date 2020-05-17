from self_play import concat_hisotry
import sys

if __name__ == '__main__':
    do_concat = len(sys.argv) > 1 and sys.argv[1] == 'concat'
    if do_concat:
        concat_hisotry()
    else:
        print("do nothing")
