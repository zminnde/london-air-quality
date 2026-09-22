#ifndef SRC_PYTHON_FUNCTION_H_
#define SRC_PYTHON_FUNCTION_H_

#include <iostream>
#include <cstdio>
#include <cstdlib>

#include <sys/wait.h>
#include <unistd.h>

#include "Function.h"

#define PYTHON_INTERPRETER "/Users/mziemys/PycharmProjects/LondonAirQuality/.venv/bin/python"
#define PYTHON_WORKER_SCRIPT "/Users/mziemys/PycharmProjects/LondonAirQuality/scripts/worker.py"

class PythonFunction: public Function {

private:
	int pipe_cpp_to_python[2];
	int pipe_python_to_cpp[2];

	FILE* write_to_python;
	FILE* read_from_python;

	pid_t python_pid;

protected:

	void init_python_function_call() {
		if (pipe(pipe_cpp_to_python) == -1 || pipe(pipe_python_to_cpp) == -1) {
			std::cerr << "Failed to create pipes.\n";
			exit(1);
		}

		python_pid = fork();

		if (python_pid == -1) {
			std::cerr << "Failed to fork process.\n";
			exit(1);
		}

		if (python_pid == 0) {

			dup2(pipe_cpp_to_python[0], STDIN_FILENO);
			dup2(pipe_python_to_cpp[1], STDOUT_FILENO);

			close(pipe_cpp_to_python[0]); close(pipe_cpp_to_python[1]);
			close(pipe_python_to_cpp[0]); close(pipe_python_to_cpp[1]);

			execlp(PYTHON_INTERPRETER,
			       PYTHON_INTERPRETER,
			       "-u",
			       PYTHON_WORKER_SCRIPT,
			       nullptr);

			std::cerr << "Failed to launch Python interpreter.\n";
			exit(1);

		} else {

			close(pipe_cpp_to_python[0]);
			close(pipe_python_to_cpp[1]);

			write_to_python = fdopen(pipe_cpp_to_python[1], "w");
			read_from_python = fdopen(pipe_python_to_cpp[0], "r");

			return;
		}
	}

	double* evaluate_func_values(double* decision_point){

		double * func_values= new double[m];

		fprintf(write_to_python, "%.17g %.17g\n", decision_point[0], decision_point[1]);
		fflush(write_to_python);

		double f1, f2;
		if (fscanf(read_from_python, "%lf %lf", &f1, &f2) == 2) {

			func_values[0]=f1;
			func_values[1]=f2;

		} else {
			std::cout<<"ERROR while getting values from Python function"<<std::endl;
			exit(1);
		}

		return func_values;
	}

public:

	PythonFunction()  {
		d=2;//  dimension of decision space
		m=2;//  number of objective functions

		double upper_bound[]={ 3,  4}; // decision  space bounds
		double lower_bound[]={-3, -4}; // decision  space bounds

		intailize(d, upper_bound, lower_bound);

		init_python_function_call();
	}

	~PythonFunction() {
		fclose(write_to_python);
		fclose(read_from_python);

		waitpid(python_pid, nullptr, 0);
	}
};

#endif /* SRC_PYTHON_FUNCTION_H_ */