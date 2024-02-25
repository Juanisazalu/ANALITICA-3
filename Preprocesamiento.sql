DROP TABLE IF EXISTS general_data2;
CREATE TABLE general_data2 AS
SELECT * FROM general_data
WHERE  strftime('%Y',InfoDate) = '2015' 
;

DROP TABLE IF EXISTS encuesta_empleado2;
CREATE TABLE encuesta_empleado2 AS
SELECT * FROM encuesta_empleado
WHERE strftime('%Y',DateSurvey) = '2015'
;

DROP TABLE IF EXISTS encuesta_gerente2;
CREATE TABLE encuesta_gerente2 AS
SELECT * FROM encuesta_gerente
WHERE strftime('%Y',SurveyDate) = '2015'
;

DROP TABLE IF EXISTS info_retiros2 ;
CREATE TABLE info_retiros2 AS
SELECT * FROM info_retiros
WHERE strftime('%Y',retirementDate) = '2015'
;

DROP TABLE IF EXISTS tabla_completa;
CREATE TABLE  tabla_completa AS
SELECT * FROM general_data2 AS gd
LEFT JOIN encuesta_empleado2 as ee ON gd.EmployeeID = ee.EmployeeID
LEFT JOIN encuesta_gerente2 as eg ON eg.EmployeeID = gd.EmployeeID
LEFT JOIN info_retiros2 AS ir ON ir.EmployeeID = eg.EmployeeID
;