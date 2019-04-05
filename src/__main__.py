# -*- coding: utf-8 -*-
from data.dataset import Dataset

def main():
    """Main function.
    """
    
    ds = Dataset([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
    print("head(3):", ds.head(3))
    print("tail(3):", ds.tail(3))
    print("split(7):", ds.split(7))
    print("split(0.7):", ds.split(0.7))
    print("[0]", ds[0])
    print("shuffle()")
    ds.shuffle()
    print("[0]", ds[0])

if __name__ == "__main__":
    main()
