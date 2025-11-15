use data_project;
select * from hr_data;

-- Sum of total salary
create table TotalSalary as
select sum(salary) from Hr_data;

Create table SalaryPosition as 
select Position, sum(Salary) as SumSalary
from Hr_data
group by Position
order by SumSalary;

-- Department
DROP TABLE IF EXISTS DepEmp;
CREATE TABLE DepEmp AS
-- number of employee per department
SELECT Department, COUNT(*) AS TotalEmployee
FROM hr_data
GROUP BY Department;
select * from DepEmp;

---- average salary per department
CREATE TABLE DepartmentSalaryAvg (
    Department VARCHAR(50),
    AvgSalary FLOAT
);
insert into DepartmentSalaryAvg(Department, AvgSalary)
select Department, AVG(Salary) 
from hr_data
group by Department;
select * from DepartmentSalaryAvg;


--- Demographic Analysis
create table MaritalDescribe as
select MaritalDesc,count(*)
from hr_data
group by MaritalDesc;
select * from MaritalDescribe;


create table Gender as
select Sex,count(*)
from hr_data
group by Sex;
select * from Gender;

-- Employment Status & Termination Analysis
-- Active vs. terminated employees
select * from Hr_data;
create table  EmploymentStatus as
select EmploymentStatus, count(*)
from Hr_data
group by EmploymentStatus;


create table TerminationForCauseReason as 
select EmploymentStatus,EmpStatusID  ,TermReason, count(*) EmpNum
from Hr_data
where EmploymentStatus="Terminated for Cause"
group by TermReason,EmploymentStatus,EmpStatusID  ;

create table VoluntarilyTerminationReason as 
select EmploymentStatus,EmpStatusID  ,TermReason, count(*) EmpNum
from Hr_data
where EmploymentStatus="Voluntarily Terminated"
group by TermReason,EmploymentStatus,EmpStatusID  ;


-- Attendance Analysis
--- Late arrivals or absences
create table AbsencesEmployee as
select Employee_Name, Absences
from Hr_data
where Absences>8;

create table AbsencesEmployee as
select Employee_Name, Absences
from Hr_data
where Absences>8;


-- Performance Employee

create table PerformanceEmployee as
select PerformanceScore, count(*) EmployeeNum
from Hr_data
group by PerformanceScore;

--- performance score based on department
create table PerformanceDepartmentEmployee as
select Department, PerformanceScore, count(*)
from Hr_data
group by PerformanceScore,Department
order by Department,PerformanceScore;

-- Select employees with a bad performance score and too many absences
create table BadPerformanceAbsence as
select Employee_Name, Absences, PerformanceScore
from Hr_data
where Absences>10 and PerformanceScore="Needs Improvement";

