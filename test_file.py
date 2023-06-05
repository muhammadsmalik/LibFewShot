if __name__ == "__main__":

    f = open('test.csv', 'r')
    content = f.read()
    print(content, flush=True)
    f.close()