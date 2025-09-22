#pragma once
#include <condition_variable>
#include <iostream>
using std::condition_variable;
using std::cout;
using std::endl;
using std::flush;
using std::max;
using std::mutex;
using std::string;
using std::unique_lock;

class Logger
{
private:
    mutex locker;
    condition_variable CV;
    int maxLine;

public:
    Logger()
    {
        this->maxLine = 0;
        cout << endl
             << flush;
    }
    ~Logger()
    {
        // show cursor
        cout << "\033[?25h" << flush;
    }

    void log(int line, string message)
    {
        unique_lock<mutex> lock(this->locker);

        // making new lines
        if (line > this->maxLine)
        {
            for (int i = 0; i < line - this->maxLine; i++)
                cout << "\n";

            cout << "\033[" << line - this->maxLine << "A";
        }

        // go back up maxLine
        cout << "\033[" << this->maxLine + 1 << "A";
        // go down line, start at column 1, and clear
        cout << "\033[" << line << "B\033[1G\033[2K";
        // print message
        cout << message;
        this->maxLine = max(this->maxLine, line);
        // go up line, and go down new maxLine
        cout << "\033[" << line << "A\033[" << this->maxLine + 1 << "B\033[1G" << flush;
    }
};